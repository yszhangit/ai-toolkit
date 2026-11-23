"""Non-blocking remote ComfyUI preview sampler for ai-toolkit.

When Machine A finishes saving a LoRA checkpoint, this module enqueues a
background preview job without pausing training. A worker thread submits the
user's exported ComfyUI API workflow JSON to Machine B over LAN, rewrites any
SaveImage / PreviewImage node to ComfyUI's built-in SaveImageWebsocket node so
nothing is written to Machine B's filesystem, streams the resulting PNG bytes
back over WebSocket, and saves them atomically into Machine A's job `samples/`
folder (with `.thumbs/`) so the ai-toolkit Web UI displays them automatically.

Uses only Python's standard library + `requests` + `Pillow` (no external
websocket library required).
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import os
import queue
import socket
import ssl
import struct
import threading
import time
import urllib.parse
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import requests
from PIL import Image

if TYPE_CHECKING:
    from toolkit.config_modules import GenerateImageConfig


class RemoteComfyUIConfig:
    """Configuration for non-blocking remote ComfyUI preview generation."""

    def __init__(self, **kwargs):
        self.enabled: bool = kwargs.get('enabled', False)
        self.server_url: str = kwargs.get('server_url', 'http://127.0.0.1:8188')
        self.workflow_path: Optional[str] = kwargs.get('workflow_path', None)
        # Path prefix inside ComfyUI's models/loras/ where Machine A's output folder is mounted (e.g. via sshfs)
        self.remote_lora_prefix: str = kwargs.get('remote_lora_prefix', '')
        # Optional local root on Machine A to compute relative LoRA path from (defaults to job's training_folder)
        self.local_root: Optional[str] = kwargs.get('local_root', None)
        # Path separator used by ComfyUI on Machine B ("/" for Linux/macOS, "\\" for Windows)
        self.path_separator: str = kwargs.get('path_separator', '/')
        # Automatically convert SaveImage / PreviewImage nodes to SaveImageWebsocket so 0 files hit Machine B's disk
        self.replace_save_with_websocket: bool = kwargs.get('replace_save_with_websocket', True)
        # Optional overrides applied to workflow JSON before placeholder substitution:
        #   {"10.inputs.lora_name": "{{lora_name}}", "6.inputs.text": "{{prompt}}"}
        self.node_overrides: dict = kwargs.get('node_overrides', {}) or {}
        # Trigger remote preview on every checkpoint save (if False, only on save steps divisible by sample_every)
        self.sample_on_save: bool = kwargs.get('sample_on_save', True)
        # Skip local GPU sampling on Machine A so training never pauses
        self.skip_local_sample: bool = kwargs.get('skip_local_sample', True)
        # Wait for queued remote previews to finish streaming back when training completes
        self.wait_on_finish: bool = kwargs.get('wait_on_finish', True)
        self.wait_timeout: float = float(kwargs.get('wait_timeout', 1800.0))
        self.request_timeout: float = float(kwargs.get('request_timeout', 600.0))
        # Retries when ComfyUI rejects a newly saved LoRA due to sshfs directory-cache lag
        self.lora_wait_retries: int = int(kwargs.get('lora_wait_retries', 6))
        self.lora_wait_interval: float = float(kwargs.get('lora_wait_interval', 5.0))

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_BINARY_EVENT_PREVIEW_IMAGE = 1
_BINARY_EVENT_PREVIEW_IMAGE_WITH_METADATA = 4
_SAVE_NODE_CLASSES = ("SaveImage", "PreviewImage")


class RemoteComfyUIError(RuntimeError):
    """Raised when a remote ComfyUI prompt submission or WebSocket read fails."""


class _StdlibWebSocket:
    """Minimal RFC 6455 WebSocket client using only Python's standard library.

    Supports ws:// and wss://, fragmented messages (FIN=0 + continuation),
    16-bit/64-bit extended payload lengths (needed for multi-MB PNGs), and
    control frames (PING/PONG/CLOSE).
    """

    def __init__(self, ws_url: str, timeout: float = 600.0):
        self.ws_url = ws_url
        self.timeout = float(timeout)
        self._sock: Optional[socket.socket] = None
        self._buf = bytearray()

    def connect(self) -> None:
        parsed = urllib.parse.urlparse(self.ws_url)
        if parsed.scheme not in ("ws", "wss"):
            raise RemoteComfyUIError(f"Unsupported WebSocket scheme: {parsed.scheme!r}")
        host = parsed.hostname
        if not host:
            raise RemoteComfyUIError(f"Missing hostname in WebSocket URL: {self.ws_url!r}")
        use_tls = parsed.scheme == "wss"
        port = parsed.port or (443 if use_tls else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        raw_sock = socket.create_connection((host, port), timeout=self.timeout)
        raw_sock.settimeout(self.timeout)
        if use_tls:
            ctx = ssl.create_default_context()
            sock: socket.socket = ctx.wrap_socket(raw_sock, server_hostname=host)
        else:
            sock = raw_sock

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        host_header = f"{host}:{port}" if parsed.port else host
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        ).encode("ascii")
        sock.sendall(req)

        header_buf = bytearray()
        while b"\r\n\r\n" not in header_buf:
            chunk = sock.recv(4096)
            if not chunk:
                sock.close()
                raise RemoteComfyUIError("WebSocket closed during HTTP upgrade handshake")
            header_buf.extend(chunk)
            if len(header_buf) > 65536:
                sock.close()
                raise RemoteComfyUIError("WebSocket HTTP upgrade response header too large")

        sep = header_buf.index(b"\r\n\r\n")
        header_bytes = bytes(header_buf[:sep])
        leftover = header_buf[sep + 4 :]

        lines = header_bytes.decode("iso-8859-1").split("\r\n")
        status_line = lines[0] if lines else ""
        if " 101 " not in status_line:
            sock.close()
            raise RemoteComfyUIError(f"WebSocket handshake rejected: {status_line}")

        headers: Dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        expected_accept = base64.b64encode(
            hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if headers.get("sec-websocket-accept") != expected_accept:
            sock.close()
            raise RemoteComfyUIError("WebSocket Sec-WebSocket-Accept mismatch")

        self._sock = sock
        self._buf = bytearray(leftover)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._send_frame(0x8, b"")
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __enter__(self) -> "_StdlibWebSocket":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _read_exact(self, n: int) -> bytes:
        assert self._sock is not None
        while len(self._buf) < n:
            chunk = self._sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                raise EOFError("WebSocket connection closed unexpectedly")
            self._buf.extend(chunk)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        if self._sock is None:
            return
        mask_key = os.urandom(4)
        header = bytearray([0x80 | (opcode & 0x0F)])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        header.extend(mask_key)
        masked = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
        self._sock.sendall(bytes(header) + masked)

    def _read_frame(self) -> Tuple[bool, int, bytes]:
        b1, b2 = self._read_exact(2)
        fin = bool(b1 & 0x80)
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        mask_key = self._read_exact(4) if masked else None
        payload = self._read_exact(length) if length > 0 else b""
        if mask_key is not None and payload:
            payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
        return fin, opcode, payload

    def recv(self) -> Optional[Any]:
        """Receive one complete WebSocket message (str for text, bytes for binary, None on close)."""
        message_opcode: Optional[int] = None
        fragments: List[bytes] = []

        while True:
            try:
                fin, opcode, payload = self._read_frame()
            except EOFError:
                return None

            # Control frames may be interleaved between fragments.
            if opcode == 0x8:  # CLOSE
                try:
                    self._send_frame(0x8, b"")
                except Exception:
                    pass
                return None
            if opcode == 0x9:  # PING
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:  # PONG
                continue

            if opcode in (0x1, 0x2):
                message_opcode = opcode
                fragments = [payload]
            elif opcode == 0x0:  # CONTINUATION
                if message_opcode is None:
                    raise RemoteComfyUIError("Unexpected WebSocket continuation frame")
                fragments.append(payload)
            else:
                continue

            if fin:
                full = b"".join(fragments)
                if message_opcode == 0x1:
                    return full.decode("utf-8", errors="replace")
                return full


def http_to_ws_url(server_url: str, client_id: str) -> str:
    base = server_url.strip().rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme == "http":
        ws_scheme = "ws"
    elif parsed.scheme == "https":
        ws_scheme = "wss"
    elif parsed.scheme in ("ws", "wss"):
        ws_scheme = parsed.scheme
    else:
        raise RemoteComfyUIError(f"Unsupported ComfyUI server_url scheme: {server_url!r}")
    path = parsed.path.rstrip("/") + "/ws"
    query = urllib.parse.urlencode({"clientId": client_id})
    return urllib.parse.urlunparse((ws_scheme, parsed.netloc, path, "", query, ""))


def resolve_remote_lora_name(
    local_lora_path: str,
    local_root: str,
    remote_lora_prefix: str = "",
    path_separator: str = "/",
) -> str:
    """Map Machine A's checkpoint path to Machine B's ComfyUI models/loras relative name."""
    abs_lora = os.path.abspath(local_lora_path)
    abs_root = os.path.abspath(local_root)
    rel = os.path.relpath(abs_lora, abs_root)
    if rel.startswith(".."):
        raise RemoteComfyUIError(
            f"Saved LoRA path {abs_lora!r} is outside local_root {abs_root!r}; "
            "configure sample.remote_comfyui.local_root to the directory mounted via sshfs."
        )
    parts: List[str] = []
    prefix = (remote_lora_prefix or "").strip().strip("/\\")
    if prefix:
        parts.extend(p for p in prefix.replace("\\", "/").split("/") if p)
    parts.extend(p for p in rel.replace("\\", "/").split("/") if p and p != ".")
    sep = path_separator or "/"
    return sep.join(parts)


def apply_node_overrides(workflow: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply YAML node_overrides onto a copy of a ComfyUI API workflow dict.

    Supports both:
      - dotted paths: {"10.inputs.lora_name": "{{lora_name}}"}
      - nested dicts: {"10": {"lora_name": "{{lora_name}}"}}
    """
    wf = copy.deepcopy(workflow)
    if not overrides:
        return wf

    for key, value in overrides.items():
        if isinstance(value, dict) and "." not in key:
            node_id = str(key)
            if node_id not in wf:
                raise RemoteComfyUIError(f"node_overrides references unknown node ID {node_id!r}")
            inputs = wf[node_id].setdefault("inputs", {})
            for field, field_val in value.items():
                inputs[field] = copy.deepcopy(field_val)
            continue

        parts = str(key).split(".")
        if len(parts) < 2:
            raise RemoteComfyUIError(
                f"Invalid node_overrides key {key!r}; expected '<node_id>.inputs.<field>' or nested dict"
            )
        node_id = parts[0]
        if node_id not in wf:
            raise RemoteComfyUIError(f"node_overrides references unknown node ID {node_id!r}")
        target = wf[node_id]
        for seg in parts[1:-1]:
            if seg not in target or not isinstance(target[seg], dict):
                target[seg] = {}
            target = target[seg]
        target[parts[-1]] = copy.deepcopy(value)
    return wf


def render_workflow_template(workflow: Any, variables: Dict[str, Any]) -> Any:
    """Recursively substitute {{var}} placeholders, preserving numeric types on exact matches."""
    if isinstance(workflow, dict):
        return {k: render_workflow_template(v, variables) for k, v in workflow.items()}
    if isinstance(workflow, list):
        return [render_workflow_template(v, variables) for v in workflow]
    if isinstance(workflow, str):
        for name, val in variables.items():
            token = "{{" + name + "}}"
            if workflow == token:
                return val
        out = workflow
        for name, val in variables.items():
            token = "{{" + name + "}}"
            if token in out:
                out = out.replace(token, "" if val is None else str(val))
        return out
    return workflow


def rewrite_save_nodes_to_websocket(
    workflow: Dict[str, Any], replace_save: bool = True
) -> Tuple[Dict[str, Any], Set[str]]:
    """Ensure the workflow outputs images via SaveImageWebsocket instead of writing to disk.

    Returns (rewritten_workflow, set_of_save_image_websocket_node_ids).
    """
    wf = copy.deepcopy(workflow)
    ws_nodes: Set[str] = {
        str(nid)
        for nid, node in wf.items()
        if isinstance(node, dict) and node.get("class_type") == "SaveImageWebsocket"
    }

    if replace_save:
        for nid, node in list(wf.items()):
            if not isinstance(node, dict):
                continue
            cls = node.get("class_type")
            if cls in _SAVE_NODE_CLASSES:
                inputs = node.get("inputs") or {}
                if "images" not in inputs:
                    continue
                wf[nid] = {
                    "class_type": "SaveImageWebsocket",
                    "inputs": {"images": inputs["images"]},
                }
                ws_nodes.add(str(nid))

    return wf, ws_nodes


def extract_ws_image_bytes(frame: bytes) -> Optional[bytes]:
    """Extract raw image bytes from a ComfyUI binary WebSocket frame."""
    if len(frame) < 8:
        return None
    event_type = struct.unpack(">I", frame[:4])[0]
    if event_type == _BINARY_EVENT_PREVIEW_IMAGE:
        return frame[8:]
    if event_type == _BINARY_EVENT_PREVIEW_IMAGE_WITH_METADATA:
        meta_len = struct.unpack(">I", frame[4:8])[0]
        start = 8 + meta_len
        if start < len(frame):
            return frame[start:]
    return None


def _is_lora_not_in_list_error(status_code: int, body_text: str) -> bool:
    if status_code != 400:
        return False
    lower = body_text.lower()
    return "value_not_in_list" in lower and ("lora" in lower or ".safetensors" in lower)


def execute_comfyui_workflow_ws(
    server_url: str,
    workflow: Dict[str, Any],
    ws_node_ids: Set[str],
    request_timeout: float = 600.0,
    lora_wait_retries: int = 6,
    lora_wait_interval: float = 5.0,
    log_fn: Callable[[str], None] = print,
) -> List[bytes]:
    """Submit a workflow to ComfyUI and stream back PNG bytes from SaveImageWebsocket nodes."""
    if not ws_node_ids:
        raise RemoteComfyUIError(
            "Workflow has no SaveImageWebsocket node (and no SaveImage/PreviewImage node to rewrite)."
        )

    client_id = uuid.uuid4().hex
    ws_url = http_to_ws_url(server_url, client_id)
    prompt_url = server_url.strip().rstrip("/") + "/prompt"
    payload = {"prompt": workflow, "client_id": client_id}

    with _StdlibWebSocket(ws_url, timeout=request_timeout) as ws:
        prompt_id: Optional[str] = None
        max_attempts = max(1, int(lora_wait_retries) + 1)
        for attempt in range(1, max_attempts + 1):
            resp = requests.post(prompt_url, json=payload, timeout=30.0)
            if resp.status_code == 200:
                data = resp.json()
                prompt_id = data.get("prompt_id")
                if not prompt_id:
                    raise RemoteComfyUIError(f"ComfyUI /prompt returned no prompt_id: {data}")
                break
            body = resp.text
            if _is_lora_not_in_list_error(resp.status_code, body) and attempt < max_attempts:
                log_fn(
                    f"[RemoteComfyUI] Machine B has not seen the new LoRA yet via sshfs "
                    f"(attempt {attempt}/{max_attempts}); retrying in {lora_wait_interval:.1f}s..."
                )
                time.sleep(lora_wait_interval)
                continue
            raise RemoteComfyUIError(
                f"ComfyUI /prompt failed (HTTP {resp.status_code}): {body[:1000]}"
            )

        assert prompt_id is not None
        current_node: Optional[str] = None
        images: List[bytes] = []

        while True:
            msg = ws.recv()
            if msg is None:
                raise RemoteComfyUIError(
                    f"ComfyUI WebSocket closed before prompt {prompt_id} completed"
                )
            if isinstance(msg, str):
                try:
                    event = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                ev_type = event.get("type")
                ev_data = event.get("data") or {}
                ev_pid = ev_data.get("prompt_id")
                if ev_pid is not None and ev_pid != prompt_id:
                    continue

                if ev_type == "executing":
                    node = ev_data.get("node")
                    if node is None:
                        break
                    current_node = str(node)
                elif ev_type == "execution_success":
                    break
                elif ev_type == "execution_error":
                    exc_msg = ev_data.get("exception_message") or str(ev_data)
                    node_id = ev_data.get("node_id")
                    node_type = ev_data.get("node_type")
                    raise RemoteComfyUIError(
                        f"ComfyUI execution error on node {node_id} ({node_type}): {exc_msg}"
                    )
                elif ev_type == "execution_interrupted":
                    raise RemoteComfyUIError(f"ComfyUI prompt {prompt_id} was interrupted")
            elif isinstance(msg, (bytes, bytearray)):
                if current_node in ws_node_ids:
                    img_bytes = extract_ws_image_bytes(bytes(msg))
                    if img_bytes:
                        images.append(img_bytes)

        return images


class RemoteComfyUISampler:
    """Background FIFO worker that dispatches checkpoint previews to a LAN ComfyUI machine."""

    def __init__(
        self,
        config: RemoteComfyUIConfig,
        training_folder: str,
        job_name: str,
        log_fn: Callable[[str], None] = print,
    ):
        self.config = config
        self.job_name = job_name
        self.local_root = os.path.abspath(config.local_root or training_folder)
        self.log_fn = log_fn

        if not config.workflow_path:
            raise ValueError("sample.remote_comfyui.workflow_path must be set when remote_comfyui.enabled is true")
        if not os.path.isfile(config.workflow_path):
            raise FileNotFoundError(f"Remote ComfyUI workflow file not found: {config.workflow_path}")

        self._workflow_template: Dict[str, Any] = self._load_workflow_file()
        self._queue: "queue.Queue[Optional[Tuple[int, str, List[GenerateImageConfig]]]]" = queue.Queue()
        self._pending_lock = threading.Lock()
        self._pending_counts: Dict[str, int] = {}
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="RemoteComfyUISampler",
            daemon=True,
        )
        self._thread.start()
        self.log_fn(
            f"[RemoteComfyUI] Enabled -> {self.config.server_url} "
            f"(workflow={self.config.workflow_path}, local_root={self.local_root}, "
            f"remote_lora_prefix={self.config.remote_lora_prefix!r})"
        )

    def _load_workflow_file(self) -> Dict[str, Any]:
        with open(self.config.workflow_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data:
            raise RemoteComfyUIError(
                f"Workflow file {self.config.workflow_path!r} must be a ComfyUI API-format JSON object "
                "(export via Workflow -> Export (API))."
            )
        if "nodes" in data and isinstance(data["nodes"], list):
            raise RemoteComfyUIError(
                f"Workflow file {self.config.workflow_path!r} looks like a UI graph JSON, not API format. "
                "In ComfyUI, use Workflow -> Export (API) to export the API-format JSON."
            )
        return data

    def _get_latest_workflow(self) -> Dict[str, Any]:
        try:
            wf = self._load_workflow_file()
            self._workflow_template = wf
        except Exception as e:
            self.log_fn(f"[RemoteComfyUI] Warning: failed to reload workflow ({e}); using cached copy")
        return self._workflow_template

    def get_pending_lora_paths(self) -> Set[str]:
        """Return absolute paths of checkpoints still queued or actively rendering on Machine B."""
        with self._pending_lock:
            return {p for p, cnt in self._pending_counts.items() if cnt > 0}

    def enqueue(
        self,
        step: int,
        local_lora_path: str,
        image_configs: List[GenerateImageConfig],
    ) -> None:
        if not image_configs:
            return
        abs_lora = os.path.abspath(local_lora_path)
        with self._pending_lock:
            self._pending_counts[abs_lora] = self._pending_counts.get(abs_lora, 0) + 1
        self._queue.put((int(step), abs_lora, list(image_configs)))
        self.log_fn(
            f"[RemoteComfyUI] Queued {len(image_configs)} sample(s) for step {step} "
            f"({os.path.basename(abs_lora)}) without pausing training."
        )

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all queued remote previews to finish. Returns True if drained before timeout."""
        limit = self.config.wait_timeout if timeout is None else timeout
        if self._queue.unfinished_tasks == 0:
            return True
        self.log_fn(
            f"[RemoteComfyUI] Waiting for {self._queue.unfinished_tasks} pending checkpoint preview "
            f"batch(es) from {self.config.server_url} (timeout={limit:.0f}s)..."
        )
        deadline = time.monotonic() + max(0.0, float(limit))
        while self._queue.unfinished_tasks > 0 and self._thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.log_fn(
                    f"[RemoteComfyUI] Timed out waiting for remote previews "
                    f"({self._queue.unfinished_tasks} batch(es) still unfinished)."
                )
                return False
            time.sleep(min(0.5, remaining))
        return self._queue.unfinished_tasks == 0

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            step, abs_lora, image_configs = item
            try:
                self._process_batch(step, abs_lora, image_configs)
            except Exception as e:
                self.log_fn(f"[RemoteComfyUI] Error rendering step {step} ({os.path.basename(abs_lora)}): {e}")
            finally:
                with self._pending_lock:
                    cnt = self._pending_counts.get(abs_lora, 1) - 1
                    if cnt <= 0:
                        self._pending_counts.pop(abs_lora, None)
                    else:
                        self._pending_counts[abs_lora] = cnt
                self._queue.task_done()

    def _process_batch(
        self,
        step: int,
        abs_lora: str,
        image_configs: List[GenerateImageConfig],
    ) -> None:
        lora_name = resolve_remote_lora_name(
            abs_lora,
            self.local_root,
            remote_lora_prefix=self.config.remote_lora_prefix,
            path_separator=self.config.path_separator,
        )
        base_wf = self._get_latest_workflow()
        overridden_wf = apply_node_overrides(base_wf, self.config.node_overrides)
        ws_wf, ws_node_ids = rewrite_save_nodes_to_websocket(
            overridden_wf, replace_save=self.config.replace_save_with_websocket
        )

        total = len(image_configs)
        saved_count = 0
        t0 = time.monotonic()

        for idx, gen_cfg in enumerate(image_configs):
            if self._stop_event.is_set():
                break
            variables = {
                "prompt": gen_cfg.prompt or "",
                "negative_prompt": gen_cfg.negative_prompt or "",
                "lora_name": lora_name,
                "lora_strength": float(gen_cfg.network_multiplier),
                "seed": int(gen_cfg.seed),
                "steps": int(gen_cfg.num_inference_steps),
                "cfg": float(gen_cfg.guidance_scale),
                "width": int(gen_cfg.width),
                "height": int(gen_cfg.height),
                "step": int(step),
                "job_name": self.job_name,
            }
            rendered_wf = render_workflow_template(ws_wf, variables)
            try:
                raw_images = execute_comfyui_workflow_ws(
                    server_url=self.config.server_url,
                    workflow=rendered_wf,
                    ws_node_ids=ws_node_ids,
                    request_timeout=self.config.request_timeout,
                    lora_wait_retries=self.config.lora_wait_retries,
                    lora_wait_interval=self.config.lora_wait_interval,
                    log_fn=self.log_fn,
                )
            except Exception as e:
                self.log_fn(
                    f"[RemoteComfyUI] Step {step} sample {idx + 1}/{total} failed: {e}"
                )
                continue

            if not raw_images:
                self.log_fn(
                    f"[RemoteComfyUI] Step {step} sample {idx + 1}/{total} returned 0 images over WebSocket."
                )
                continue

            pil_img = Image.open(io.BytesIO(raw_images[0]))
            pil_img.load()
            gen_cfg.save_image_atomic(pil_img, count=idx, max_count=max(0, total - 1))
            saved_count += 1

        elapsed = time.monotonic() - t0
        self.log_fn(
            f"[RemoteComfyUI] Step {step}: streamed {saved_count}/{total} sample(s) "
            f"from {self.config.server_url} in {elapsed:.1f}s (lora={lora_name})"
        )


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Test streaming a preview image from a remote ComfyUI server via SaveImageWebsocket."
    )
    parser.add_argument("--server", required=True, help="ComfyUI server URL, e.g. http://192.168.1.50:8188")
    parser.add_argument("--workflow", required=True, help="Path to ComfyUI API-format workflow.json")
    parser.add_argument("--lora", required=True, help="LoRA name as seen by ComfyUI on Machine B")
    parser.add_argument("--prompt", default="a portrait photo", help="Positive prompt")
    parser.add_argument("--neg", default="", help="Negative prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--output", default="./remote_comfyui_test.png", help="Output path on Machine A")
    args = parser.parse_args()

    with open(args.workflow, "r", encoding="utf-8") as f:
        wf = json.load(f)
    ws_wf, ws_nodes = rewrite_save_nodes_to_websocket(wf, replace_save=True)
    rendered = render_workflow_template(
        ws_wf,
        {
            "prompt": args.prompt,
            "negative_prompt": args.neg,
            "lora_name": args.lora,
            "lora_strength": args.strength,
            "seed": args.seed,
            "steps": args.steps,
            "cfg": args.cfg,
            "width": args.width,
            "height": args.height,
            "step": 0,
            "job_name": "test",
        },
    )
    imgs = execute_comfyui_workflow_ws(args.server, rendered, ws_nodes)
    if not imgs:
        raise SystemExit("No images received from SaveImageWebsocket.")
    out_img = Image.open(io.BytesIO(imgs[0]))
    out_img.save(args.output)
    print(f"Saved streamed image ({out_img.size[0]}x{out_img.size[1]}) to {args.output}")


if __name__ == "__main__":
    _cli()

"""Unit and end-to-end tests for toolkit.remote_comfyui_sampler using a mock ComfyUI HTTP+WS server."""

import base64
import hashlib
import io
import json
import os
import socket
import struct
import tempfile
import threading
import time
import unittest
from typing import Dict, List, Tuple

from PIL import Image

try:
    from toolkit.remote_comfyui_sampler import (
        RemoteComfyUIConfig,
        RemoteComfyUIError,
        RemoteComfyUISampler,
        apply_node_overrides,
        bind_titled_nodes,
        execute_comfyui_workflow_ws,
        extract_ws_image_bytes,
        find_titled_node_bindings,
        render_workflow_template,
        resolve_remote_lora_name,
        rewrite_save_nodes_to_websocket,
    )
except ImportError:
    import importlib.util

    _mod_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "toolkit",
        "remote_comfyui_sampler.py",
    )
    _spec = importlib.util.spec_from_file_location("remote_comfyui_sampler", _mod_path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    RemoteComfyUIConfig = _mod.RemoteComfyUIConfig
    RemoteComfyUIError = _mod.RemoteComfyUIError
    RemoteComfyUISampler = _mod.RemoteComfyUISampler
    apply_node_overrides = _mod.apply_node_overrides
    bind_titled_nodes = _mod.bind_titled_nodes
    execute_comfyui_workflow_ws = _mod.execute_comfyui_workflow_ws
    extract_ws_image_bytes = _mod.extract_ws_image_bytes
    find_titled_node_bindings = _mod.find_titled_node_bindings
    render_workflow_template = _mod.render_workflow_template
    resolve_remote_lora_name = _mod.resolve_remote_lora_name
    rewrite_save_nodes_to_websocket = _mod.rewrite_save_nodes_to_websocket

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _make_png_bytes(rgb: Tuple[int, int, int], size: Tuple[int, int] = (32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, rgb).save(buf, format="PNG")
    return buf.getvalue()


def _ws_server_frame(opcode: int, payload: bytes, fin: bool = True) -> bytes:
    b1 = (0x80 if fin else 0x00) | (opcode & 0x0F)
    n = len(payload)
    if n < 126:
        header = bytes([b1, n])
    elif n < 65536:
        header = struct.pack("!BBH", b1, 126, n)
    else:
        header = struct.pack("!BBQ", b1, 127, n)
    return header + payload


class _MockComfyServer:
    """Minimal mock ComfyUI server speaking HTTP (/prompt, /system_stats, /object_info) and RFC 6455 /ws."""

    def __init__(
        self,
        fail_first_n_not_in_list: int = 0,
        use_metadata_event: bool = False,
        loras_root: str = "",
        loras_prefix: str = "",
        extra_object_info: Dict = None,
        missing_classes: set = None,
    ):
        self.fail_first_n = fail_first_n_not_in_list
        self.use_metadata_event = use_metadata_event
        self.loras_root = loras_root
        self.loras_prefix = loras_prefix.strip("/\\")
        self.extra_object_info = extra_object_info or {}
        self.missing_classes = set(missing_classes or ())
        self.submitted_workflows: List[Dict] = []
        self._lock = threading.Lock()
        self._ws_clients: Dict[str, socket.socket] = {}
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self.port = self._sock.getsockname()[1]
        self._sock.listen(16)
        self._sock.settimeout(0.2)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _scan_loras(self) -> List[str]:
        if not self.loras_root or not os.path.isdir(self.loras_root):
            return []
        found = []
        for root, _, files in os.walk(self.loras_root):
            for fn in files:
                if fn.endswith(".safetensors"):
                    rel = os.path.relpath(os.path.join(root, fn), self.loras_root).replace("\\", "/")
                    found.append(f"{self.loras_prefix}/{rel}" if self.loras_prefix else rel)
        return sorted(found)

    def _object_info_for(self, cls: str) -> Dict:
        if cls in self.missing_classes:
            return {}
        if cls in self.extra_object_info:
            return {cls: self.extra_object_info[cls]}
        if cls in ("LoraLoader", "LoraLoaderModelOnly"):
            return {
                cls: {
                    "input": {
                        "required": {
                            "lora_name": [self._scan_loras()],
                            "strength_model": ["FLOAT", {"default": 1.0}],
                        }
                    }
                }
            }
        if cls == "SaveImageWebsocket":
            return {cls: {"input": {"required": {"images": ["IMAGE"]}}}}
        if cls == "UNETLoader":
            return {cls: {"input": {"required": {"unet_name": [["model_a.safetensors"]], "weight_dtype": [["default"]]}}}}
        if cls == "QwenImage21Cache":
            return {cls: {"input": {"required": {"device": ["COMBO", {"options": ["auto", "cpu"]}], "dtype": ["COMBO", {"options": ["default"]}]}}}}
        return {cls: {"input": {"required": {}}}}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except Exception:
            pass
        self._thread.join(timeout=2.0)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

    def _handle_conn(self, conn: socket.socket) -> None:
        conn.settimeout(5.0)
        try:
            raw = bytearray()
            while b"\r\n\r\n" not in raw:
                chunk = conn.recv(4096)
                if not chunk:
                    conn.close()
                    return
                raw.extend(chunk)
            sep = raw.index(b"\r\n\r\n")
            head = raw[:sep].decode("iso-8859-1")
            body = bytearray(raw[sep + 4 :])
            lines = head.split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            if target.startswith("/ws"):
                client_id = target.split("clientId=", 1)[1].split("&", 1)[0]
                accept = base64.b64encode(
                    hashlib.sha1((headers["sec-websocket-key"] + _WS_GUID).encode("ascii")).digest()
                ).decode("ascii")
                resp = (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n"
                    "\r\n"
                ).encode("ascii")
                conn.sendall(resp)
                with self._lock:
                    self._ws_clients[client_id] = conn
                # Keep socket open until client closes or server stops.
                while not self._stop.is_set():
                    time.sleep(0.05)
                return

            if method == "GET" and target == "/system_stats":
                ok_body = json.dumps(
                    {
                        "system": {"comfyui_version": "0.45.0-mock"},
                        "devices": [{"name": "cuda:0 Mock GPU"}],
                    }
                ).encode("utf-8")
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(ok_body)}\r\nConnection: close\r\n\r\n".encode("ascii")
                    + ok_body
                )
                conn.close()
                return

            if method == "GET" and target.startswith("/object_info/"):
                cls = target[len("/object_info/") :].split("?", 1)[0]
                ok_body = json.dumps(self._object_info_for(cls)).encode("utf-8")
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(ok_body)}\r\nConnection: close\r\n\r\n".encode("ascii")
                    + ok_body
                )
                conn.close()
                return

            if method == "POST" and target == "/prompt":
                content_len = int(headers.get("content-length", "0"))
                while len(body) < content_len:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    body.extend(chunk)
                req = json.loads(bytes(body[:content_len]).decode("utf-8"))
                client_id = req["client_id"]
                wf = req["prompt"]

                with self._lock:
                    if self.fail_first_n > 0:
                        self.fail_first_n -= 1
                        err_body = json.dumps(
                            {
                                "error": {"type": "prompt_outputs_failed_validation"},
                                "node_errors": {
                                    "10": {
                                        "errors": [
                                            {
                                                "type": "value_not_in_list",
                                                "details": "lora_name: 'x.safetensors' not in list",
                                            }
                                        ]
                                    }
                                },
                            }
                        ).encode("utf-8")
                        conn.sendall(
                            b"HTTP/1.1 400 Bad Request\r\n"
                            b"Content-Type: application/json\r\n"
                            + f"Content-Length: {len(err_body)}\r\nConnection: close\r\n\r\n".encode("ascii")
                            + err_body
                        )
                        conn.close()
                        return
                    self.submitted_workflows.append(wf)
                    pid = f"pid-{len(self.submitted_workflows)}"
                    ws_conn = self._ws_clients.get(client_id)

                ok_body = json.dumps({"prompt_id": pid}).encode("utf-8")
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(ok_body)}\r\nConnection: close\r\n\r\n".encode("ascii")
                    + ok_body
                )
                conn.close()

                if ws_conn is not None:
                    self._emit_ws_run(ws_conn, pid, wf)
                return
        except Exception:
            try:
                conn.close()
            except Exception:
                pass

    def _emit_ws_run(self, ws_conn: socket.socket, pid: str, wf: Dict) -> None:
        ws_node = next(nid for nid, node in wf.items() if node.get("class_type") == "SaveImageWebsocket")
        # 1. Send a PING to verify client PONG handling.
        ws_conn.sendall(_ws_server_frame(0x9, b"hi"))
        # 2. Simulate KSampler executing and emitting an intermediate preview frame (MUST be ignored!).
        ws_conn.sendall(
            _ws_server_frame(0x1, json.dumps({"type": "executing", "data": {"node": "3", "prompt_id": pid}}).encode())
        )
        bogus_preview = struct.pack(">II", 1, 1) + _make_png_bytes((255, 0, 0), (16, 16))
        ws_conn.sendall(_ws_server_frame(0x2, bogus_preview))

        # 3. Execute SaveImageWebsocket node and stream a fragmented PNG!
        ws_conn.sendall(
            _ws_server_frame(0x1, json.dumps({"type": "executing", "data": {"node": ws_node, "prompt_id": pid}}).encode())
        )
        seed = int(wf.get("3", {}).get("inputs", {}).get("seed", 0))
        color = ((seed * 37) % 256, 128, 64)
        png = _make_png_bytes(color, (48, 32))
        if self.use_metadata_event:
            meta = json.dumps({"image_type": "image/png"}).encode("utf-8")
            full_bin = struct.pack(">II", 4, len(meta)) + meta + png
        else:
            full_bin = struct.pack(">II", 1, 2) + png

        # Split binary message into 3 WebSocket fragments to exercise continuation frames.
        cut1 = len(full_bin) // 3
        cut2 = (2 * len(full_bin)) // 3
        ws_conn.sendall(_ws_server_frame(0x2, full_bin[:cut1], fin=False))
        ws_conn.sendall(_ws_server_frame(0x0, full_bin[cut1:cut2], fin=False))
        ws_conn.sendall(_ws_server_frame(0x0, full_bin[cut2:], fin=True))

        # 4. Signal execution completion.
        ws_conn.sendall(
            _ws_server_frame(0x1, json.dumps({"type": "executing", "data": {"node": None, "prompt_id": pid}}).encode())
        )


class _DummyGenConfig:
    """Lightweight stand-in for GenerateImageConfig for standalone unit testing without torch."""

    def __init__(self, out_dir: str, step: int, prompt: str, seed: int):
        self.out_dir = out_dir
        self.step = step
        self.prompt = prompt
        self.negative_prompt = "blurry"
        self.network_multiplier = 0.85
        self.seed = seed
        self.num_inference_steps = 25
        self.guidance_scale = 3.0
        self.width = 1024
        self.height = 768
        self.saved_paths: List[str] = []

    def save_image_atomic(self, image: Image.Image, count: int = 0, max_count: int = 0) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"step_{self.step:09d}_{count}.png")
        image.save(path)
        self.saved_paths.append(path)


class TestRemoteComfyUISampler(unittest.TestCase):
    def test_resolve_remote_lora_name(self):
        self.assertEqual(
            resolve_remote_lora_name(
                "/data/output/my_job/my_job_000000200.safetensors",
                "/data/output",
                remote_lora_prefix="aitk_output",
                path_separator="/",
            ),
            "aitk_output/my_job/my_job_000000200.safetensors",
        )
        self.assertEqual(
            resolve_remote_lora_name(
                "/data/output/my_job/my_job_000000200.safetensors",
                "/data/output",
                remote_lora_prefix="aitk_output",
                path_separator="\\",
            ),
            "aitk_output\\my_job\\my_job_000000200.safetensors",
        )
        with self.assertRaises(RemoteComfyUIError):
            resolve_remote_lora_name("/other/file.safetensors", "/data/output")

    def test_overrides_template_and_ws_rewrite(self):
        wf = {
            "3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 7.0}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
            "10": {"class_type": "LoraLoader", "inputs": {"lora_name": "old.safetensors", "strength_model": 1.0}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}},
        }
        wf2 = apply_node_overrides(
            wf,
            {
                "10.inputs.lora_name": "{{lora_name}}",
                "10.inputs.strength_model": "{{lora_strength}}",
                "6.inputs.text": "prefix {{prompt}}",
                "3": {"seed": "{{seed}}", "steps": "{{steps}}", "cfg": "{{cfg}}"},
            },
        )
        ws_wf, ws_nodes = rewrite_save_nodes_to_websocket(wf2, replace_save=True)
        self.assertEqual(ws_nodes, {"9"})
        self.assertEqual(ws_wf["9"]["class_type"], "SaveImageWebsocket")
        self.assertEqual(ws_wf["9"]["inputs"], {"images": ["8", 0]})

        rendered = render_workflow_template(
            ws_wf,
            {
                "lora_name": "aitk/job/job_000000200.safetensors",
                "lora_strength": 0.85,
                "prompt": "a cat",
                "seed": 42,
                "steps": 25,
                "cfg": 3.0,
            },
        )
        self.assertEqual(rendered["10"]["inputs"]["lora_name"], "aitk/job/job_000000200.safetensors")
        self.assertIs(type(rendered["10"]["inputs"]["strength_model"]), float)
        self.assertAlmostEqual(rendered["10"]["inputs"]["strength_model"], 0.85)
        self.assertEqual(rendered["6"]["inputs"]["text"], "prefix a cat")
        self.assertIs(type(rendered["3"]["inputs"]["seed"]), int)
        self.assertEqual(rendered["3"]["inputs"]["seed"], 42)

    def test_extract_ws_image_bytes(self):
        png = _make_png_bytes((1, 2, 3))
        frame1 = struct.pack(">II", 1, 2) + png
        self.assertEqual(extract_ws_image_bytes(frame1), png)
        meta = b'{"image_type":"image/png"}'
        frame4 = struct.pack(">II", 4, len(meta)) + meta + png
        self.assertEqual(extract_ws_image_bytes(frame4), png)

    def test_end_to_end_streaming_with_sshfs_retry_and_fragmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = os.path.join(tmp, "output")
            job_dir = os.path.join(output_root, "my_job")
            samples_dir = os.path.join(job_dir, "samples")
            os.makedirs(job_dir, exist_ok=True)
            lora_file = os.path.join(job_dir, "my_job_000000200.safetensors")
            with open(lora_file, "wb") as f:
                f.write(b"fake")

            server = _MockComfyServer(
                fail_first_n_not_in_list=2,
                use_metadata_event=False,
                loras_root=output_root,
                loras_prefix="aitk_output",
            )
            try:
                wf_path = os.path.join(tmp, "workflow_api.json")
                with open(wf_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "3": {"class_type": "KSampler", "inputs": {"seed": "{{seed}}", "steps": "{{steps}}", "cfg": "{{cfg}}"}},
                            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}"}},
                            "10": {"class_type": "LoraLoader", "inputs": {"lora_name": "{{lora_name}}", "strength_model": "{{lora_strength}}"}},
                            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}},
                        },
                        f,
                    )

                cfg = RemoteComfyUIConfig(
                    enabled=True,
                    server_url=server.url,
                    workflow_path=wf_path,
                    remote_lora_prefix="aitk_output",
                    lora_wait_retries=3,
                    lora_wait_interval=0.05,
                    request_timeout=10.0,
                    preflight_check="strict",
                    preflight_sshfs_probe=True,
                )
                logs: List[str] = []
                sampler = RemoteComfyUISampler(
                    cfg,
                    training_folder=output_root,
                    job_name="my_job",
                    log_fn=logs.append,
                    save_root=job_dir,
                    sample_prompts_count=2,
                )
                try:
                    self.assertTrue(sampler.preflight_report["ok"])
                    # Ensure sentinel probe file was cleaned up immediately.
                    self.assertEqual([fn for fn in os.listdir(job_dir) if "_aitk_remote_probe_" in fn], [])
                    c0 = _DummyGenConfig(samples_dir, 200, "portrait 1", seed=42)
                    c1 = _DummyGenConfig(samples_dir, 200, "portrait 2", seed=43)
                    sampler.enqueue(200, lora_file, [c0, c1])
                    self.assertTrue(sampler.wait_for_completion(timeout=10.0))
                    self.assertEqual(sampler.get_pending_lora_paths(), set())
                finally:
                    sampler.stop()

                self.assertEqual(len(server.submitted_workflows), 2)
                self.assertEqual(
                    server.submitted_workflows[0]["10"]["inputs"]["lora_name"],
                    "aitk_output/my_job/my_job_000000200.safetensors",
                )
                # Verify both images arrived from SaveImageWebsocket (not the 16x16 KSampler preview!).
                for idx, (c, seed) in enumerate(((c0, 42), (c1, 43))):
                    self.assertEqual(len(c.saved_paths), 1)
                    img = Image.open(c.saved_paths[0])
                    self.assertEqual(img.size, (48, 32))
                    self.assertEqual(img.getpixel((0, 0)), ((seed * 37) % 256, 128, 64))
            finally:
                server.close()

    def test_preflight_catches_errors_and_respects_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = os.path.join(tmp, "output")
            job_dir = os.path.join(output_root, "my_job")
            os.makedirs(job_dir, exist_ok=True)
            wf_path = os.path.join(tmp, "wf.json")
            with open(wf_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default"}},
                        "5": {"class_type": "UNETLoader", "inputs": {"unet_name": "missing_model.safetensors", "weight_dtype": "default"}},
                        "10": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "{{lora_name}}", "strength_model": "{{lora_strength}}"}},
                        "9": {"class_type": "SaveImageWebsocket", "inputs": {"images": ["8", 0]}},
                    },
                    f,
                )

            # Mount prefix intentionally mismatches server's prefix ("wrong_prefix" vs "aitk_output")
            server = _MockComfyServer(loras_root=output_root, loras_prefix="aitk_output")
            try:
                cfg_strict = RemoteComfyUIConfig(
                    enabled=True,
                    server_url=server.url,
                    workflow_path=wf_path,
                    remote_lora_prefix="wrong_prefix",
                    preflight_check="strict",
                    preflight_sshfs_probe=True,
                )
                with self.assertRaises(RemoteComfyUIError) as ctx:
                    RemoteComfyUISampler(
                        cfg_strict,
                        training_folder=output_root,
                        job_name="my_job",
                        log_fn=lambda _: None,
                        save_root=job_dir,
                        sample_prompts_count=0,
                    )
                err = str(ctx.exception)
                self.assertIn("samples_non_empty", err)
                self.assertIn("missing_model.safetensors", err)
                self.assertIn("sshfs_mount_probe", err)
                # Sentinel file must be cleaned up even when preflight raises!
                self.assertEqual([fn for fn in os.listdir(job_dir) if "_aitk_remote_probe_" in fn], [])

                # Same config in "warn" mode must succeed and record the errors in preflight_report.
                cfg_warn = RemoteComfyUIConfig(
                    enabled=True,
                    server_url=server.url,
                    workflow_path=wf_path,
                    remote_lora_prefix="wrong_prefix",
                    preflight_check="warn",
                    preflight_sshfs_probe=True,
                )
                sampler_warn = RemoteComfyUISampler(
                    cfg_warn,
                    training_folder=output_root,
                    job_name="my_job",
                    log_fn=lambda _: None,
                    save_root=job_dir,
                    sample_prompts_count=0,
                )
                try:
                    self.assertFalse(sampler_warn.preflight_report["ok"])
                    self.assertGreaterEqual(len(sampler_warn.preflight_report["errors"]), 3)
                finally:
                    sampler_warn.stop()

                # "off" mode skips preflight completely (even with an unreachable server URL).
                cfg_off = RemoteComfyUIConfig(
                    enabled=True,
                    server_url="http://127.0.0.1:1",
                    workflow_path=wf_path,
                    preflight_check="off",
                )
                sampler_off = RemoteComfyUISampler(
                    cfg_off,
                    training_folder=output_root,
                    job_name="my_job",
                    log_fn=lambda _: None,
                    save_root=job_dir,
                    sample_prompts_count=0,
                )
                try:
                    self.assertIsNone(sampler_off.preflight_report)
                finally:
                    sampler_off.stop()
            finally:
                server.close()

    def test_titled_node_binding_preserves_user_workflow_params(self):
        # Raw ComfyUI API export with TWO LoRAs (Turbo + training LoRA titled AITK_LORA) and AITK_PROMPT
        raw_wf = {
            "1": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 8, "cfg": 1}},
            "3": {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {"prompt": "old prompt", "negative_prompt": "keep my neg", "resolution": 1024},
                "_meta": {"title": "AITK_PROMPT"},
            },
            "11": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"lora_name": "qi21/turbo.safetensors", "strength_model": 1.0, "model": ["5", 0]},
                "_meta": {"title": "Load Turbo LoRA"},
            },
            "10": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"lora_name": "some_dummy_lora_not_on_disk.safetensors", "strength_model": 0.75, "model": ["11", 0]},
                "_meta": {"title": "AITK_LORA"},
            },
            "9": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}},
        }
        bound, bindings = bind_titled_nodes(
            raw_wf,
            lora_name="pc-a/my_job/my_job_000000250.safetensors",
            prompt="tester42, portrait",
            negative_prompt="ignored because no AITK_NEG_PROMPT",
        )
        self.assertEqual(bindings["lora"], [("10", "lora_name")])
        self.assertEqual(bindings["prompt"], [("3", "prompt")])
        # Training LoRA & prompt replaced:
        self.assertEqual(bound["10"]["inputs"]["lora_name"], "pc-a/my_job/my_job_000000250.safetensors")
        self.assertEqual(bound["3"]["inputs"]["prompt"], "tester42, portrait")
        # All user-chosen workflow params stay 100% untouched:
        self.assertEqual(bound["11"]["inputs"]["lora_name"], "qi21/turbo.safetensors")
        self.assertEqual(bound["10"]["inputs"]["strength_model"], 0.75)
        self.assertEqual(bound["3"]["inputs"]["negative_prompt"], "keep my neg")
        self.assertEqual(bound["1"]["inputs"]["steps"], 8)
        self.assertEqual(bound["1"]["inputs"]["cfg"], 1)

        # Preflight must skip validating placeholder lora_name on AITK_LORA while STILL validating Turbo LoRA on "11"
        with tempfile.TemporaryDirectory() as tmp:
            output_root = os.path.join(tmp, "output")
            job_dir = os.path.join(output_root, "my_job")
            os.makedirs(job_dir, exist_ok=True)
            turbo_file = os.path.join(output_root, "turbo.safetensors")
            with open(turbo_file, "wb") as f:
                f.write(b"x")
            raw_wf["11"]["inputs"]["lora_name"] = "pc-a/turbo.safetensors"
            wf_path = os.path.join(tmp, "raw_8steps.json")
            with open(wf_path, "w", encoding="utf-8") as f:
                json.dump(raw_wf, f)

            server = _MockComfyServer(loras_root=output_root, loras_prefix="pc-a")
            try:
                cfg = RemoteComfyUIConfig(
                    enabled=True,
                    server_url=server.url,
                    workflow_path=wf_path,
                    remote_lora_prefix="pc-a",
                    preflight_check="strict",
                    preflight_sshfs_probe=True,
                )
                sampler = RemoteComfyUISampler(
                    cfg,
                    training_folder=output_root,
                    job_name="my_job",
                    log_fn=lambda _: None,
                    save_root=job_dir,
                    sample_prompts_count=1,
                )
                try:
                    self.assertTrue(sampler.preflight_report["ok"])
                    self.assertEqual(sampler.preflight_report["warnings"], [])
                finally:
                    sampler.stop()
            finally:
                server.close()

    def test_metadata_binary_event(self):
        server = _MockComfyServer(fail_first_n_not_in_list=0, use_metadata_event=True)
        try:
            wf = {
                "3": {"class_type": "KSampler", "inputs": {"seed": 7}},
                "9": {"class_type": "SaveImageWebsocket", "inputs": {"images": ["8", 0]}},
            }
            imgs = execute_comfyui_workflow_ws(server.url, wf, {"9"}, request_timeout=10.0)
            self.assertEqual(len(imgs), 1)
            img = Image.open(io.BytesIO(imgs[0]))
            self.assertEqual(img.size, (48, 32))
            self.assertEqual(img.getpixel((0, 0)), ((7 * 37) % 256, 128, 64))
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()

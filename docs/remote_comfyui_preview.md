# Remote ComfyUI Preview Sampler (`ai-toolkit` Machine A → ComfyUI Machine B)

Generate training sample previews on a second LAN GPU (**Machine B** running ComfyUI) **without pausing training on Machine A**, streaming the generated PNG bytes back over ComfyUI's WebSocket directly into Machine A's `<output>/<job_name>/samples/` folder (zero image files written to Machine B's disk).

---

## 1. Built-in ComfyUI Node: `SaveImageWebsocket`

ComfyUI ships a built-in node called `SaveImageWebsocket` that sends generated PNG bytes over ComfyUI's `/ws?clientId=...` WebSocket binary stream instead of saving files to Machine B's `output/` directory:

- Official ComfyUI source (`SaveImageWebsocket`):
  https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_images.py
- Official ComfyUI WebSocket image streaming example (`websockets_api_example_ws_images.py`):
  https://github.com/Comfy-Org/ComfyUI/blob/master/script_examples/websockets_api_example_ws_images.py

> **Why you didn't see it in ComfyUI's node menu:** `SaveImageWebsocket` is under the `api/image` category and marked for API workflows (often hidden from normal UI search unless Dev Mode is enabled).
>
> **You do NOT need to add it manually:** `toolkit/remote_comfyui_sampler.py` has `replace_save_with_websocket: true` enabled by default—it automatically rewrites any standard `SaveImage` or `PreviewImage` node in your exported API `workflow.json` into `SaveImageWebsocket` before sending it to Machine B.

---

## 2. Machine B Setup (`sshfs` Mount)

Mount Machine A's `ai-toolkit` output directory inside Machine B's `ComfyUI/models/loras/` folder (e.g., `aitk_output`):

```bash
mkdir -p ~/ComfyUI/models/loras/aitk_output
sshfs -o cache=no,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
  yzhang@<MACHINE_A_IP>:/home/yzhang/Desktop/ai-toolkit/output \
  ~/ComfyUI/models/loras/aitk_output
```

- `-o cache=no` (or `dir_cache=no`) avoids sshfs directory-listing cache delays so ComfyUI sees new `.safetensors` checkpoints immediately.
- As an extra safeguard, `RemoteComfyUISampler` automatically retries if ComfyUI briefly returns `value_not_in_list` for a newly saved checkpoint.
- Make sure ComfyUI on Machine B listens on the LAN (`python main.py --listen 0.0.0.0 --port 8188`).

---

## 3. Exporting Your Workflow from Machine B (Zero Manual JSON Editing)

1. Build and test any workflow in ComfyUI on Machine B (standard, 8-step Turbo LoRA + training LoRA, MiniMax H3 Image, etc.) with your own `steps`, `cfg`, `seed`, `width`, `height`, `negative_prompt`, and `strength_model` — **all sampler parameters stay your responsibility in the workflow**.
2. In ComfyUI's UI, right-click and **rename the Title** of just 2 nodes:
   - Your training LoRA loader node (`LoraLoaderModelOnly` / `LoraLoader`) → **`AITK_LORA`**
     *(Any other LoRA nodes, such as a fixed Turbo LoRA, keep their normal title and are left untouched.)*
   - Your prompt / text-encode node (`TextEncodeQwenImage21`, `CLIPTextEncode`, etc.) → **`AITK_PROMPT`**
     *(Replaces `prompt` or `text` with each prompt from `sample.samples`; leaves `negative_prompt` untouched unless you optionally title a node `AITK_NEG_PROMPT`.)*
3. Leave your normal `SaveImage` or `PreviewImage` node as-is (`replace_save_with_websocket: true` automatically converts it to `SaveImageWebsocket` on submission).
4. Export **API format**: **Workflow → Export (API)** (not regular Save), copy the `.json` to Machine A, and point `workflow_path` at it — **no manual JSON editing required**.

*(Optional advanced fallback: `{{lora_name}}`, `{{lora_strength}}`, `{{prompt}}`, `{{negative_prompt}}`, `{{seed}}`, `{{steps}}`, `{{cfg}}`, `{{width}}`, `{{height}}` placeholders and `node_overrides` are still supported if you ever want YAML-driven overrides.)*

---

## 4. `ai-toolkit` Job YAML Configuration

Add `remote_comfyui:` under `sample:` in your job YAML:

```yaml
sample:
  sampler: flowmatch
  sample_every: 200
  width: 1024
  height: 1024
  samples:
    - prompt: "triggerword, portrait photo..."
    - prompt: "triggerword, three-quarter view..."
  neg: ""
  seed: 42
  walk_seed: true
  guidance_scale: 3.0
  sample_steps: 25
  remote_comfyui:
    enabled: true
    server_url: "http://pc-b:8188"
    workflow_path: "./docs/workflows/lora_tester_qi21_api.json"
    remote_lora_prefix: "aitk_output"
    # local_root: "/home/yzhang/Desktop/ai-toolkit/output"  # defaults to job training_folder
    path_separator: "/"                # use "\\" if Machine B runs Windows ComfyUI
    replace_save_with_websocket: true  # rewrites SaveImage/PreviewImage -> SaveImageWebsocket
    sample_on_save: true               # queue remote preview on every checkpoint save
    skip_local_sample: true            # never pause Machine A for local sampling
    wait_on_finish: true               # wait for queued remote previews when training ends
    wait_timeout: 1800
    request_timeout: 600
    lora_wait_retries: 6
    lora_wait_interval: 5.0
    lora_node_title: "AITK_LORA"       # ComfyUI node title whose lora_name gets replaced with each checkpoint
    prompt_node_title: "AITK_PROMPT"   # ComfyUI node title whose prompt/text gets replaced per sample prompt
    neg_prompt_node_title: "AITK_NEG_PROMPT"  # optional; leave unused to keep workflow's own negative prompt
    preflight_check: "strict"          # "strict" (fail fast on startup), "warn" (log & continue), or "off"
    preflight_timeout: 10.0            # seconds per HTTP/WS startup check
    preflight_sshfs_probe: true        # write+delete 0-byte sentinel .safetensors to verify sshfs & prefix
    # Optional if you prefer NOT to edit workflow.json with {{...}} placeholders:
    node_overrides:
      "10.inputs.lora_name": "{{lora_name}}"
      "10.inputs.strength_model": "{{lora_strength}}"
      "6.inputs.text": "{{prompt}}"
      "7.inputs.text": "{{negative_prompt}}"
      "3.inputs.seed": "{{seed}}"
      "3.inputs.steps": "{{steps}}"
      "3.inputs.cfg": "{{cfg}}"
      "5.inputs.width": "{{width}}"
      "5.inputs.height": "{{height}}"
```

---

## 5. Startup Preflight Checks

When `sample.remote_comfyui.enabled: true` and `preflight_check` is `"strict"` (default) or `"warn"`, `ai-toolkit` runs fast (< 1s, no GPU render) startup checks before training begins:

1. **`samples_non_empty`** — Verifies `sample.samples` has at least 1 prompt (catches YAML indentation errors immediately).
2. **`node_overrides`** — Verifies every node ID in `node_overrides` exists in `workflow_path`.
3. **`ws_output_node`** — Verifies at least one `SaveImageWebsocket` (or `SaveImage` / `PreviewImage` when `replace_save_with_websocket: true`) node exists.
4. **`lora_placeholder`** — Warns if `{{lora_name}}` does not appear anywhere in the workflow/overrides.
5. **`lora_path_mapping`** — Verifies `save_root` sits under `local_root` and logs the resolved remote `lora_name` pattern.
6. **`server_reachable`** — Calls `GET {server_url}/system_stats` and logs Machine B's ComfyUI version & GPUs.
7. **`websocket_handshake`** — Opens and closes `/ws?clientId=...` to verify WebSocket upgrade works.
8. **`workflow_nodes` & `workflow_models`** — Calls `GET {server_url}/object_info/{class_type}` for `SaveImageWebsocket` and every node in the workflow, verifying no missing nodes and that static model/combo inputs (`unet_name`, `clip_name`, `vae_name`, `sampler_name`, etc.) exist on Machine B.
9. **`sshfs_mount_probe`** (when `preflight_sshfs_probe: true`) — Writes a 0-byte sentinel `_aitk_remote_probe_<id>.safetensors` in `save_root`, checks `GET {server_url}/object_info/LoraLoaderModelOnly` for the mapped `lora_name`, and deletes the sentinel file immediately. This verifies `sshfs`, `-o cache=no`, and `remote_lora_prefix` end-to-end.

---

## 6. Standalone Smoke Test & Preflight CLI (Before Training)

Run startup preflight checks only (< 1s, no render):

```bash
python3 /home/yzhang/Desktop/ai-toolkit/toolkit/remote_comfyui_sampler.py \
  --preflight \
  --server http://pc-b:8188 \
  --workflow ./docs/workflows/lora_tester_qi21_api.json \
  --local-root /home/yzhang/ai-toolkit/output \
  --save-root /home/yzhang/ai-toolkit/output/qi21_lora_test \
  --prefix pc-a
```

Run a full 1-image WebSocket streaming test:

You can test Machine A → Machine B streaming directly from the CLI:

```bash
python3 -m toolkit.remote_comfyui_sampler \
  --server http://pc-b:8188 \
  --workflow ./docs/workflows/lora_tester_qi21_api.json \
  --lora mine/qi21/tz_qi21_74img_768_autov3.safetensors \
  --prompt "a portrait of woman" \
  --output ./test_preview.png
```

An example Qwen Image 2.1 API workflow with `LoraLoaderModelOnly` wired between `UNETLoader` and `QwenImage21Cache` and `SaveImageWebsocket` as the output node is included at `docs/workflows/lora_tester_qi21_api.json`.

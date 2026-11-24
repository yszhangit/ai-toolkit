# Remote ComfyUI Preview Sampler (`ai-toolkit` Machine A → ComfyUI Machine B)

Generate training sample previews on a second LAN GPU (**Machine B** running ComfyUI) **without pausing training on Machine A**, streaming the generated PNG bytes back over ComfyUI's WebSocket directly into Machine A's `<output>/<job_name>/samples/` folder (zero image files written to Machine B's disk).

When `sample.remote_comfyui.enabled: true`, **all local sample settings** (`steps`, `cfg`, `seed`, `width`, `height`, `neg`, `sampler`, `network_multiplier`, `walk_seed`) are **ignored** except `sample.samples` (the prompt list). Your ComfyUI workflow owns all sampling parameters; `ai-toolkit` touches only **2 inputs + 1 output**:

1. **Input 1 (`AITK_LORA`):** Sets `lora_name` to the newly saved checkpoint on Machine B.
2. **Input 2 (`AITK_PROMPT`):** Sets `prompt` / `text` to each prompt in `sample.samples`.
3. **Output (`SaveImage` / `PreviewImage` → `SaveImageWebsocket`):** Streams PNG bytes back over `/ws` with zero files written on Machine B.

---

## 1. Built-in ComfyUI Node: `SaveImageWebsocket`

ComfyUI ships a built-in `SaveImageWebsocket` node that streams PNG bytes over `/ws?clientId=...` instead of writing to disk:

- Official ComfyUI source (`SaveImageWebsocket`):
  https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_images.py
- Official WebSocket image-streaming example (`websockets_api_example_ws_images.py`):
  https://github.com/Comfy-Org/ComfyUI/blob/master/script_examples/websockets_api_example_ws_images.py

> You do **not** need to add `SaveImageWebsocket` manually in ComfyUI's UI. `replace_save_with_websocket: true` automatically rewrites any standard `SaveImage` or `PreviewImage` node in your exported API workflow JSON.

---

## 2. Machine B Setup (`sshfs` Mount)

Mount Machine A's `ai-toolkit` output directory inside Machine B's `ComfyUI/models/loras/` folder (e.g., `pc-a`):

```bash
mkdir -p ~/ComfyUI/models/loras/pc-a
sshfs -o cache=no,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
  yzhang@pc-a:/home/yzhang/ai-toolkit/output \
  ~/ComfyUI/models/loras/pc-a
```

- `-o cache=no` avoids sshfs directory-cache lag so ComfyUI sees new `.safetensors` checkpoints immediately.
- As an extra safeguard, `RemoteComfyUISampler` automatically retries if ComfyUI briefly returns `value_not_in_list`.
- Start ComfyUI on Machine B listening on the LAN (`python main.py --listen 0.0.0.0 --port 8188`).

---

## 3. Exporting Your Workflow from Machine B (Zero Manual JSON Editing)

1. Build and test any workflow in ComfyUI on Machine B (standard, 8-step Turbo LoRA + training LoRA, MiniMax H3 Image, etc.) with all your own sampler settings (`steps`, `cfg`, `seed`, `width`, `height`, `negative_prompt`, `strength_model`).
2. In ComfyUI's UI, right-click and **rename the Title** of **2 nodes**:
   - Your training LoRA loader node (`LoraLoaderModelOnly` / `LoraLoader`) → **`AITK_LORA`**
     *(Other LoRA nodes such as a fixed Turbo LoRA keep their normal title and are left untouched.)*
   - Your prompt / text-encode node (`TextEncodeQwenImage21`, `CLIPTextEncode`, etc.) → **`AITK_PROMPT`**
     *(Replaces only `prompt` or `text`; your workflow's `negative_prompt` is never modified.)*
3. Keep your normal `PreviewImage` or `SaveImage` output node.
4. Click **Workflow → Export (API)**, copy the `.json` to Machine A, and point `workflow_path` at it.

An example Qwen Image 2.1 workflow exported in this format is included at `docs/workflows/lora_tester_qi21_api.json`.

---

## 4. `ai-toolkit` Job YAML Configuration

Add `remote_comfyui:` inside `sample:` in your job YAML:

```yaml
    sample:
      sample_every: 250
      samples:
        - prompt: "tester42, portrait photo of Asian woman, natural expression"
        - prompt: "tester42, three-quarter view of Asian woman looks afar."
      remote_comfyui:
        enabled: true
        server_url: "http://pc-b:8188"
        workflow_path: "./docs/workflows/lora_tester_qi21_api.json"
        remote_lora_prefix: "pc-a"
        # local_root: "/home/yzhang/ai-toolkit/output"  # defaults to job training_folder
        path_separator: "/"                # use "\\" if Machine B runs Windows ComfyUI
        lora_node_title: "AITK_LORA"       # ComfyUI node title for training LoRA
        prompt_node_title: "AITK_PROMPT"   # ComfyUI node title for sample prompt
        replace_save_with_websocket: true  # rewrites SaveImage/PreviewImage -> SaveImageWebsocket
        sample_on_save: true               # queue remote preview on every checkpoint save
        skip_local_sample: true            # never pause Machine A for local sampling
        wait_on_finish: true               # wait for queued remote previews when training ends
        wait_timeout: 1800
        request_timeout: 600
        lora_wait_retries: 6
        lora_wait_interval: 5.0
        preflight_check: "strict"          # "strict" (fail fast), "warn" (log & continue), or "off"
        preflight_timeout: 10.0
        preflight_sshfs_probe: true        # write+delete 0-byte sentinel .safetensors to verify sshfs & prefix
```

---

## 5. Startup Preflight Checks

When `sample.remote_comfyui.enabled: true` and `preflight_check` is `"strict"` (default) or `"warn"`, `ai-toolkit` validates the **2 inputs, 1 output, Machine B, and sshfs mount** in < 1s before training starts:

1. **`samples_non_empty`** — Verifies `sample.samples` has at least 1 prompt.
2. **`lora_input_node`** — Verifies a node titled `AITK_LORA` with a `lora_name` input exists.
3. **`prompt_input_node`** — Verifies a node titled `AITK_PROMPT` with a `prompt`/`text` input exists.
4. **`ws_output_node`** — Verifies a `SaveImageWebsocket` (or rewritable `SaveImage`/`PreviewImage`) output node exists.
5. **`lora_path_mapping`** — Verifies `save_root` sits under `local_root` and logs the mapped `lora_name` pattern.
6. **`server_reachable`** — Calls `GET {server_url}/system_stats`.
7. **`websocket_handshake`** — Opens and closes `/ws?clientId=...`.
8. **`workflow_nodes` & `workflow_models`** — Verifies all workflow node classes and static model/combo inputs (`unet_name`, `clip_name`, `vae_name`, fixed Turbo LoRA, etc.) exist on Machine B (skipping the placeholder `lora_name` on `AITK_LORA`).
9. **`sshfs_mount_probe`** — Writes a 0-byte `_aitk_remote_probe_<id>.safetensors` in `save_root`, confirms Machine B sees the mapped `lora_name`, and deletes it immediately.

---

## 6. Workflow Validate / Convert Utility & Smoke Test CLI

If you export a raw ComfyUI API workflow without renaming nodes or with the training LoRA node omitted/disabled, use `scripts/comfyui_workflow_tool.py` to **validate** or **convert** it into a new file (the original file is never modified):

```bash
# 1. Inspect & validate any workflow (add --server http://pc-b:8188 to also check Machine B nodes & models):
python3 scripts/comfyui_workflow_tool.py validate user_upload/lora_tester_qi21_4steps.json \
  --server http://pc-b:8188

# 2. Convert a raw API workflow -> renames prompt to AITK_PROMPT, inserts/renames AITK_LORA,
#    rewrites PreviewImage/SaveImage -> SaveImageWebsocket, and validates against Machine B:
python3 scripts/comfyui_workflow_tool.py convert user_upload/lora_tester_qi21_4steps.json \
  -o work/lora_tester_qi21_4steps_aitk.json \
  --server http://pc-b:8188
```

- If the workflow already contains a **Turbo / Lightning / LCM LoRA** (like `lora_tester_qi21_8steps.json`), `convert` automatically keeps the Turbo LoRA untouched and stacks a new `AITK_LORA` node right after it (or pass `--insert-lora` explicitly).

Run startup preflight checks directly via the sampler CLI (< 1s, no GPU render):

```bash
python3 /home/yzhang/Desktop/ai-toolkit/toolkit/remote_comfyui_sampler.py \
  --preflight \
  --server http://pc-b:8188 \
  --workflow ./docs/workflows/lora_tester_qi21_api.json \
  --local-root /home/yzhang/ai-toolkit/output \
  --save-root /home/yzhang/ai-toolkit/output/qi21_lora_test \
  --prefix pc-a
```

Run a 1-image WebSocket streaming smoke test:

```bash
python3 /home/yzhang/Desktop/ai-toolkit/toolkit/remote_comfyui_sampler.py \
  --server http://pc-b:8188 \
  --workflow ./docs/workflows/lora_tester_qi21_api.json \
  --lora mine/qi21/tz_qi21_74img_768_autov3.safetensors \
  --prompt "a portrait of woman" \
  --output ./work/pc_b_live_test.png
```

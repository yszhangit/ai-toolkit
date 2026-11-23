# AGENTS.md — Development Guide for This `ai-toolkit` Fork

## 1. Goal

This repository is a personal fork of [`ostris/ai-toolkit`](https://github.com/ostris/ai-toolkit) maintained for:

1. **Quality-of-life training features** — such as non-blocking LAN remote ComfyUI preview generation ([docs/remote_comfyui_preview.md](docs/remote_comfyui_preview.md)), where a training machine streams preview images back from a second LAN GPU machine over WebSocket without pausing training or writing files to the preview machine's disk.
2. **Training recipe / adapter / optimizer research & selective ports** — comparing reference trainers (such as sibling repo `/home/yzhang/Desktop/Fizgig`) against `ai-toolkit` for models like **Qwen Image 2.1** ([docs/fizgig_vs_aitk_qi21.md](docs/fizgig_vs_aitk_qi21.md)) and **MiniMax H3 Image** ([docs/fizgig_vs_aitk_h3_image.md](docs/fizgig_vs_aitk_h3_image.md)), and adopting better settings, adapters, or targeted code ports while staying on `ai-toolkit`.

Training itself is executed from a **separate `ai-toolkit` installation**; changes developed here are committed to feature branches and pulled into the training installation via the fork.

---

## 2. Git Remotes & Branching Conventions

- `origin`: `https://github.com/yszhangit/ai-toolkit.git` (personal fork)
- `upstream`: `https://github.com/ostris/ai-toolkit.git`
- Always create a **dedicated feature branch** off `main` for code and documentation changes (never commit directly to `main`).
- Keep commits scoped strictly to the feature's code, tests, and `docs/` files. Do **not** stage untracked local scratch/personal files (`user_upload/`, `work/`, ad-hoc root `.yaml`/`.md` notes) unless explicitly requested.
- Do **not** push to `origin` unless asked — the repo owner (`yzhang`) pushes with their Git credentials.

---

## 3. Dev Environment, Accounts & File Permissions

- **Accounts:**
  - The AI agent runs as Linux user `aiagent` (group `developers`).
  - The repo owner is `yzhang` (also in group `developers`).
- **Workspace paths:**
  - Primary repo: `/home/yzhang/Desktop/ai-toolkit`
  - Sibling reference repo (read-only reference): `/home/yzhang/Desktop/Fizgig`
- **Sharing files with `yzhang`:**
  - **`yzhang` cannot open files inside `/home/aiagent/` directly.**
  - Always place shareable documentation and workflow templates under `/home/yzhang/Desktop/ai-toolkit/docs/`.
  - Always place temporary test artifacts meant for inspection (e.g., live preview PNGs) under `/home/yzhang/Desktop/ai-toolkit/work/` with group-readable permissions (`chmod 664`, group `developers`).
- **Permission gotchas:**
  - Existing files updated by `yzhang` (e.g., after `git pull`) may be owned `yzhang:yzhang` (`0664`), causing `permission denied` for `aiagent`. If encountered, report the exact paths to `yzhang` so they can run `chgrp developers <files>`.
  - The training Python venv (`/home/yzhang/ai-toolkit/venv`) is owned by `yzhang` and is not executable by `aiagent`. Write standalone helper modules and unit tests so they work with system `python3` (stdlib + `requests` + `Pillow`) without requiring extra `pip install`s or venv modifications.

---

## 4. Dual-Machine LAN Topology

Defined in `/etc/hosts`:

| Role | Hostnames | LAN IP | Direct Link IP | Notes |
|---|---|---|---|---|
| **Machine A (Trainer)** | `pc-a`, `yz2025l` | `192.168.1.249` | `10.0.0.2` | Runs `ai-toolkit` training jobs |
| **Machine B (Preview)** | `pc-b`, `dzhang` | `192.168.1.222` | `10.0.0.1` | Runs ComfyUI on port `8188` (`--listen 0.0.0.0`, RTX 3090 24GB + RTX 4060 Ti 16GB) |

- Machine B accesses Machine A's saved LoRA checkpoints via `sshfs` mounted into Machine B's `ComfyUI/models/loras/` directory (use `-o cache=no` to avoid directory-cache lag).
- Machine B streams preview PNGs back to Machine A over ComfyUI's built-in `SaveImageWebsocket` WebSocket protocol (zero image files written to Machine B's filesystem).

---

## 5. Documentation & Testing

- **Documentation directory:** Put all guides, model comparisons, and ComfyUI API workflow templates under `docs/`:
  - `docs/remote_comfyui_preview.md` — Remote ComfyUI preview setup & YAML reference
  - `docs/workflows/lora_tester_qi21_api.json` — Qwen Image 2.1 API workflow template
  - `docs/fizgig_vs_aitk_qi21.md` — Fizgig vs `ai-toolkit` comparison (Qwen Image 2.1)
  - `docs/fizgig_vs_aitk_h3_image.md` — Fizgig vs `ai-toolkit` comparison (MiniMax H3 Image)
- **Running unit tests (system `python3`):**
  ```bash
  cd /home/yzhang/Desktop/ai-toolkit
  python3 -m py_compile toolkit/config_modules.py jobs/process/BaseSDTrainProcess.py toolkit/remote_comfyui_sampler.py
  python3 -m unittest -v testing/test_remote_comfyui_sampler.py
  ```
- **Running a live smoke test against Machine B:**
  ```bash
  python3 /home/yzhang/Desktop/ai-toolkit/toolkit/remote_comfyui_sampler.py \
    --server http://pc-b:8188 \
    --workflow /home/yzhang/Desktop/ai-toolkit/docs/workflows/lora_tester_qi21_api.json \
    --lora mine/qi21/tz_qi21_74img_768_autov3.safetensors \
    --prompt "a portrait of woman" \
    --steps 10 --output /home/yzhang/Desktop/ai-toolkit/work/pc_b_live_test.png
  ```

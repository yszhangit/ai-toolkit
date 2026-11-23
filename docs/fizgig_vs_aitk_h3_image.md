# Fizgig vs ai-toolkit: MiniMax H3 Image LoRA Comparison

Comparison of default/recommended **MiniMax H3 Image** LoRA settings, adapters, and training mechanics between **Fizgig** (`/home/yzhang/Desktop/Fizgig`) and **ai-toolkit** (`/home/yzhang/Desktop/ai-toolkit`), with notes on what can be adopted via YAML vs ported in code.

---

## 1. Side-by-Side Comparison

| Setting / Mechanism | Fizgig Default (`Standard` / `Style`) | `ai-toolkit` UI Default (`minimax_h3_image`) |
|---|---|---|
| **Training Adapter** | **ON by default**: `ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter` (`fizgig_minimax_h3_image_training_adapter.safetensors`, r16/a16, ~150 MB, frozen @ 1.0, off in previews/save). Trained on 7.6k real photos + 1k model renders (4k steps, EMA 0.998). | **ON by default**: `ostris/minimax_h3_training_adapter/minimax_h3_image_training_adapter_v1.safetensors` (live assistant LoRA, frozen @ 1.0, off in previews/save). |
| **Rank / Alpha** | **32 / 32** (`Standard`, default), **16 / 16** (`Style`), **8 / 8** (`Edit` / `Slider`) | **16 / 16** |
| **Optimizer** | **Standard:** `automagic4` (**non-fused**, `lr=2.5e-4`, **`max_lr=2.5e-4`** cap) — Fizgig notes v4 @ `2.5e-4` beat v3 and Muon by a wide margin over ~30 runs, especially on sets with few headshots.<br>**Style / Edit / Slider:** `automagic3` (`lr=2.8e-4`, non-fused). | **adamw8bit** (`lr=1e-4`, `weight_decay=1e-4`, `eps=1e-6`).<br>*(Note: `ai-toolkit` already has `automagic4` and `automagic3` — Fizgig vendored both verbatim from `ai-toolkit`!)* |
| **Timestep Sampling (6 PDD heads)** | Both train on the exact 6 PDD sigmas `(1.0, 0.9675, 0.8875, 0.750, 0.5328, 0.2320)`, each routed to its own head.<br>**Standard has "Likeness Optimization" ON:** weights steps toward the clean-end sigma (`0.232`, where eyes/skin/likeness form):<br>• Epochs 1–6 (alternating): odd epochs ramp **16.67% → 33.33% → 50%**; even epochs **75%**.<br>• Epoch 7+: **75%** clean-end every epoch (rest split evenly across the 5 higher sigmas).<br>*(Style/Edit/Slider keep uniform 1/6.)* | **Strictly uniform (`1/6 ≈ 16.7%` each)** over `fixed_train_sigmas` (`BaseSDTrainProcess.py` draws `randint(0, 6)` and ignores `timestep_type` / `content_or_style`). |
| **LoRA Target Modules** | **200 Linears** (`blocks.0..49`: `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2`).<br>Explicitly **excludes `token_refiner`** (Fizgig notes training it softens output and causes epoch-to-epoch preview judder) and excludes `adaln_proj`. | **208 Linears**: `ignore_if_contains: ["adaln_proj"]`, and `get_transformer_block_names()` returns `["blocks"]`, which **also matches `token_refiner.blocks.0..1`** (8 extra Linears). |
| **EMA** | **ON @ 0.98** (fp32 shadow with `(1+n)/(10+n)` warmup ramp, swapped in for previews & saves) | **OFF** (`use_ema: false`, default `0.99`, no warmup ramp unless `use_num_updates=True` is passed) |
| **Resolution** | **0.25 MP** (~512×512, 32px buckets) across all presets (Fizgig explicitly tested 0.5 MP and reverted to 0.25 MP) | `[512, 768, 1024]` |
| **Base Quantization** | Defaults to `minimax_h3_image_pruned_int8_convrot.safetensors` (~21 GB) + `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (~15.7 GB); falls back to NF4 / HQQ + H2D block swap rings on smaller GPUs | Same default `convrot8` DiT + `nvfp4` TE, **plus** supports Comfy-Org's 4-bit **`convrotw4a8`** (`minimax_h3_image_pruned_w4a8.safetensors`, 12.6 GB), which Fizgig does not support |
| **LoRA Weight Dtype / Caption Dropout** | `bf16` LoRA weights (`trainable_dtype="bf16"`), caption dropout `0.0` (off by default) | `fp32` LoRA weights, caption dropout `0.05` |
| **Preview Sampler** | 6 steps, CFG 1, `2048×2048`, **LCM with decaying noise (`s_noise: 1.1 → 0.9`)** by default (notes smoother, more natural skin vs Euler's crunchiness), plus INT8 attention matching ComfyUI | 6 steps, CFG 1, `2048×2048`, **Euler** |

*(Note: Fizgig's H3 Image presets have `KREA2_PER_IMAGE_LR: True` in the dict, but `families/train.py` automatically disables per-image LR whenever `automagic3`/`automagic4` is selected because Automagic owns its own LR.)*

---

## 2. Key Findings & What Is Worth Adopting into `ai-toolkit`

### A. YAML-Only Settings You Can Apply Immediately (Zero Code Changes)
1. **Automagic v4 with `max_lr: 2.5e-4` cap (Fizgig's biggest optimizer finding for H3 Image):**
   - `ai-toolkit` already has the exact same `Automagic4` code!
   - **Critical detail:** Upstream `Automagic4` defaults to `max_lr: 1e3` (an overflow guard only). Fizgig found v4 climbs too high without a cap and explicitly sets:
     ```yaml
     optimizer: automagic4
     lr: 2.5e-4
     optimizer_params:
       max_lr: 2.5e-4
       fused: false   # keeps grad clipping (max_grad_norm) active
     ```
   - *(Note from Fizgig's docs: if you use DOP / differential output preservation, switch off Automagic to AdamW @ ~2e-4–2.5e-4, because Automagic shrinks its LR against alternating prior-preservation passes.)*
2. **Exclude `token_refiner` from LoRA targets:**
   - Change `network_kwargs.ignore_if_contains` from `["adaln_proj"]` to:
     ```yaml
     ignore_if_contains: ["adaln_proj", "token_refiner"]
     ```
   - Matches Fizgig's exact 200-Linear target set and avoids the softening / epoch-to-epoch preview judder Fizgig observed from training `token_refiner`.
3. **Rank 32 / 32 (for characters/subjects), EMA @ 0.98, and 512 resolution:**
   - Try `linear: 32, linear_alpha: 32`, `resolution: [512]` (fast, ~0.25 MP), and `ema_config: {use_ema: true, ema_decay: 0.98}`.

### B. Training Adapter: Ostris v1 vs Fizgig's Adapter
- Unlike Qwen 2.1, `ai-toolkit` **already ships** a dedicated H3 Image training adapter (`ostris/minimax_h3_training_adapter/minimax_h3_image_training_adapter_v1.safetensors`) enabled by default.
- If you want to A/B test Fizgig's adapter (`ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter`):
  - **Module shapes match 100%** (same 50 blocks × 4 fused modules: `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2`, rank 16, alpha 16).
  - **Key naming differs:** Fizgig's file uses Kohya keys (`lora_unet_blocks_0_attn_qkv_proj.lora_down.weight`), whereas `ai-toolkit`'s `load_training_adapter` (`peft_format=True`) expects ComfyUI/diffusers keys (`diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight`). Passing Fizgig's file raw will drop all 600 keys.
  - **Fix:** A trivial ~5-line key-rename script (or small check in `minimax_h3.py`'s `convert_lora_weights_before_load`) converts it losslessly with no rank/matrix changes needed.

### C. Code Ports Worth Considering for `ai-toolkit`
1. **Clean-end weighted / scheduled sampling over `fixed_train_sigmas` ("Likeness Optimization") — Highest Value Port:**
   - Currently `BaseSDTrainProcess.py` (`L1235-L1243`) hardcodes uniform `torch.randint(0, len(fixed_train_sigmas))` for H3 Image.
   - Adding optional probability weights / an epoch-or-step schedule for the last sigma (`0.232`) — e.g. ramping `16.7% → 50%` alternating with `75%` for the first ~6 epochs, then holding `75%` — is ~25 lines of code and is Fizgig's main H3 Image character-likeness secret weapon.
2. **LCM preview sampler (`s_noise: 1.1 → 0.9`):**
   - Optional ~15-line addition to `extensions_built_in/diffusion_models/minimax_h3/src/pipeline.py` if you prefer smoother skin in training previews (you can already use `SamplerLCM` in ComfyUI for final inference).
3. **EMA `use_num_updates=True`:**
   - 1-line addition in `jobs/process/BaseSDTrainProcess.py` (`setup_ema`) so early EMA checkpoints don't lag toward zero.

---

## 3. Source References

- **Fizgig:**
  - `src/fizgig/families/minimax_image.py` (presets, `LIKENESS_DEFAULTS`, training adapter, LoRA format, memory/sampler notes)
  - `src/fizgig/minimax_image/driver.py` (`_sigma`, `_clean_share`, `lora_target_names`)
  - `src/fizgig/minimax_image/sampling.py` (`LCM_NOISE`, `sample`)
  - `src/fizgig/training/optimizers.py` (`AUTOMAGIC4_MAX_LR = 2.5e-4`), `src/fizgig/training/automagic4.py`
  - `checks/memsim_headers/minimax_image_minimax_image_training_adapter.json.gz`
  - `docs/MINIMAX_IMAGE.md`
- **ai-toolkit:**
  - `extensions_built_in/diffusion_models/ui.tsx` (`minimax_h3_image`, L1571-L1671)
  - `extensions_built_in/diffusion_models/minimax_h3/minimax_h3_image.py` (`H3_IMAGE_SAMPLE_SIGMAS`, `get_train_scheduler`)
  - `extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py` (`load_training_adapter`, `get_transformer_block_names`)
  - `extensions_built_in/diffusion_models/minimax_h3/src/transformer.py` (`MiniMaxH3TokenRefiner`, `qkv_proj`, `fc1`)
  - `extensions_built_in/diffusion_models/minimax_h3/src/pipeline.py`
  - `toolkit/optimizers/automagic4.py`, `toolkit/optimizers/automagic3.py`
  - `toolkit/lora_special.py` (`peft_format`), `toolkit/models/base_model.py` (`convert_lora_weights_before_load`)
  - `jobs/process/BaseSDTrainProcess.py` (`fixed_train_sigmas` sampling, `setup_ema`)

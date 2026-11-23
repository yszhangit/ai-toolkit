# Fizgig vs ai-toolkit: Qwen Image 2.1 LoRA Comparison

Comparison of default/recommended Qwen Image 2.1 LoRA settings, adapters, and training mechanics between **Fizgig** (`/home/yzhang/Desktop/Fizgig`) and **ai-toolkit** (`/home/yzhang/Desktop/ai-toolkit`), with notes on what can be adopted via YAML vs ported in code.

---

## 1. Side-by-Side Comparison

| Setting / Mechanism | Fizgig Default (`Fast` / `Standard`) | `ai-toolkit` UI Default (`qwen_image_2`) | Existing `qi21_auto3_1024.yaml` |
|---|---|---|---|
| **Base Training Adapter** | **ON by default** on base 2.1: `ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter` (`fizgig_qwen_image_2.1_training_adapter_v2.safetensors`, r16/a16, ~84 MB, frozen @ 1.0, off for previews/save) | **OFF** on base 2.1 (UI only sets Ostris's adapter on `qwen_image_2:turbo`, though `qwen_image_2.py` supports `assistant_lora_path` on base in YAML) | None |
| **Rank / Alpha** | **8 / 8** (`Fast`, default) or **16 / 16** (`Standard` & `Style`) | **32 / 32** | 32 / 32 |
| **Optimizer** | **Muon** (`BatchedMuon`, `adjust_lr_fn=match_rms_adamw`, `weight_decay=0.01`, `momentum=0.95`, `nesterov=True`, `ns_steps=5`) | **adamw8bit** (`weight_decay=1e-4`, `eps=1e-6`) — **no Muon** in `toolkit/optimizer.py` | `automagic3` (`weight_decay=1e-4`, `fused=true`) |
| **LR / Schedule** | **Adaptive LR** (epoch-level plateau + weight-norm stability watcher w/ 70/30 rollback & learned ceiling): `2e-4..4e-4` (r8, starts `~2.83e-4`) or `1e-4..2e-4` (r16, starts `~1.41e-4`); flat `1.5e-4` for Style | Constant `1e-4` | `automagic3` starting at `1e-5` |
| **EMA** | **ON @ 0.98**, fp32 shadow with `(1+n)/(10+n)` ramp-in, swapped in for previews & checkpoint saves | **OFF** (`use_ema: false`, default `0.99`; `BaseSDTrainProcess.py` omits `use_num_updates=True`, so no early warmup ramp) | `use_ema: false` |
| **Resolution** | **0.5 MP** (~704×704 square, 32px buckets). Notes 0.5 MP keeps more sharpness than 0.25 MP and avoids softening Qwen's native skin detail toward dataset photos | `[512, 768, 1024]` | `[768, 1024]` |
| **Timestep Sampling** | **Continuous logit-normal + dynamic exponential shift** (`t = sigmoid(randn)`, then `exp(mu) / (exp(mu) + (1/t - 1))`, `base_shift=0.5 @ 256` to `max_shift=0.9 @ 8192`) | `timestep_type: shift` = **uniform** over shifted + terminal-stretched (`0.02`) 1000-step grid (not logit-normal) | `timestep_type: linear` (unshifted uniform) |
| **LoRA Target Modules** | **7 Linears × 32 blocks = 224**: unfuses `img_mlp.gate_up` into separate `img_mlp.gate_layer` + `img_mlp.proj` (`to_q, to_k, to_v, to_out.0, gate_layer, proj, out`) | **6 Linears × 32 blocks = 192**: keeps fused `img_mlp.gate_up` (`transformer_only=True`: `to_q, to_k, to_v, to_out.0, gate_up, out`) | 6 Linears × 32 blocks |
| **Base Quantization** | Auto: `bf16` on ≥24 GB VRAM (no swap), `int8` on 12–16 GB, `nf4` on 10 GB; TE unloaded after caching | Pre-quantized ComfyUI `convrot8` for DiT & TE, `low_vram: true` | `convrot8` + TE 85% offload |
| **VAE Encode / Caption Dropout** | Deterministic `latent_dist.mode()`; caption dropout `0.0` (off by default) | Stochastic `latent_dist.sample()`; caption dropout `0.05` | `.sample()`, dropout `0.05` |
| **Block Profiling / Fast ID** | Measured **blocks 10–14** carry 67–89% of character identity (block 12 strongest); optional "Fast Identity Mode" trains only blocks 10–14 (~1.5× faster) | Not a preset, but already supported via `network_kwargs.only_if_contains` | All 32 blocks |

Both trainers match on: flow-matching velocity target (`noise - x0`), fp32 LoRA weights, 32px bucket divisibility, Qwen3-VL-8B pre-RMSNorm hidden state extraction with dropped system turn, and RGBA VAE normalization.

---

## 2. Key Findings & What Is Worth Adopting into `ai-toolkit`

### A. Fizgig Base Training Adapter v2 (Highest Impact)
- **Why:** Fizgig's lab notes (`src/fizgig/families/qwen_image.py`) report that base Qwen 2.1 LoRAs wobble or collapse into texture without a frozen training adapter (measured likeness 77.4 with Fizgig's adapter vs 76.4 with SimpleTuner assistant-v2 vs 55.8 without, with noticeably better detail than SimpleTuner's). `ai-toolkit` only configures a training adapter on `qwen_image_2:turbo`, not on base `qwen_image_2`.
- **Compatibility catch (verified in code):**
  - Fizgig's `fizgig_qwen_image_2.1_training_adapter_v2.safetensors` uses **split** `img_mlp.gate_layer` and `img_mlp.proj` keys (rank 16, alpha 16; 672 tensors total).
  - `ai-toolkit`'s `QwenImage21Transformer2DModel` uses **fused** `img_mlp.gate_up` (`[gate_layer; proj]` concatenated on output dim).
  - Passing Fizgig's raw file directly to `assistant_lora_path` causes `toolkit/network_mixins.py` (`load_weights`) to **silently drop all 64 `gate_layer` and `proj` LoRA modules** under `Missing keys`.
- **Exact lossless fix:**
  - Stack `A_fused = [A_gate; A_proj]` (`[32, d]`) and block-diagonal `B_fused = diag(B_gate, B_proj)` (`[2m, 32]`) for `img_mlp.gate_up`, and zero-pad the remaining rank-16 layers (`attn.*`, `img_mlp.out`) from rank 16 → 32, setting `alpha = 32` (`32/32 = 1.0`, identical to original `16/16 = 1.0`).
  - Can be done either via a standalone offline conversion script (zero changes to `ai-toolkit`) or by adding ~20 lines in `QwenImage2Model.load_training_adapter` (`extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py`).

### B. YAML-Only Settings to Test Immediately in `ai-toolkit`
1. **Lower rank:** `linear: 16, linear_alpha: 16` (or `8 / 8` for single-character datasets) instead of `32 / 32`.
2. **EMA @ 0.98:** `ema_config: {use_ema: true, ema_decay: 0.98}`. (Optional 1-line improvement in `jobs/process/BaseSDTrainProcess.py` `setup_ema()`: pass `use_num_updates=True` so early steps don't lag behind zero init.)
3. **Resolution:** Test single-scale `resolution: [704]` or `[768]` (~0.5 MP) if 1024 softens details or slows convergence.
4. **Timesteps:** Switch `timestep_type` from `linear` to `shift` (already built in).
5. **Optional identity-block experiment:** Restrict training to `transformer_blocks.10.` through `transformer_blocks.14.` via `network_kwargs.only_if_contains` to test faster convergence / reduced style bleed.

### C. Small Code Ports Worth Considering
1. **Muon (`BatchedMuon`) optimizer:** Self-contained (~215 lines, BSD-3-Clause) in `Fizgig/src/fizgig/training/muon.py`; easy ~15-line registration in `ai-toolkit/toolkit/optimizer.py`. Uses `adjust_lr_fn=match_rms_adamw`, so AdamW learning rates apply directly.
2. **Logit-normal + dynamic shift timesteps:** ~15 lines in `ai-toolkit/toolkit/samplers/custom_flowmatch_sampler.py` to add a `logit_normal_shift` `timestep_type` matching Fizgig/SD3/Qwen training.
3. **Adaptive LR (lower priority):** `ai-toolkit` already includes `automagic3` and `automagic4` (which Fizgig vendored from `ai-toolkit`), so epoch-level `AdaptiveLR` is mainly useful only if pairing with Muon.

---

## 3. Source References

- **Fizgig:**
  - `src/fizgig/families/qwen_image.py` (presets, training adapter, LoRA format, block profiling, lab notes)
  - `src/fizgig/qwen_image21/driver.py` (`_sample_t`, `training_loss`)
  - `src/fizgig/qwen_image21/sampling.py` (`encode_image`, `calculate_mu`)
  - `src/fizgig/qwen_image21/model.py` (`SwiGLUFeedForward`)
  - `src/fizgig/training/muon.py`, `src/fizgig/training/adaptive_lr.py`, `src/fizgig/training/ema.py`
  - `checks/memsim_headers/qwen_image21_qwen21_training_adapter.json.gz`
  - `docs/QWEN_IMAGE.md`
- **ai-toolkit:**
  - `extensions_built_in/diffusion_models/ui.tsx` (`qwen_image_2`, `qwen_image_2:turbo`)
  - `ui/src/app/jobs/new/jobConfig.ts`
  - `extensions_built_in/diffusion_models/qwen_image_2/qwen_image_2.py` (`load_training_adapter`, `encode_images`)
  - `extensions_built_in/diffusion_models/qwen_image_2/src/transformer.py` (`QwenImage21SwiGLUFeedForward`)
  - `toolkit/optimizer.py`, `toolkit/ema.py`, `toolkit/network_mixins.py`, `toolkit/samplers/custom_flowmatch_sampler.py`
  - `jobs/process/BaseSDTrainProcess.py` (`setup_ema`, timestep preparation)
  - `qi21_auto3_1024.yaml`

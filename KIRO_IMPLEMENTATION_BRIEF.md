# Kiro IDE Implementation Brief — v05 Native Pipeline

**Target repo:** `BarrFarr/Auto-OpenMatte-Sdr2hdr-converter`, branch `feat/full-pipeline-implementation`
**Evidence repo:** `BarrFarr/Auto-OpenMatte-Sdr2hdr-converter--Research`
**Baseline commit at time of writing:** `d0ef113`
**Hardware assumed present:** NVIDIA RTX 3080 (GA102, 10 GB, 1 NVDEC, 1 NVENC)

This brief is derived from three diagnostics (Nsight render profile, native fit-only scaling, 16x100 px full-overlap zonal) plus the archived research reports. Every acceptance threshold below is traceable to a measured number, not to a guess.

---

## 0. System facts the agent must not re-derive

| Property | Value |
|---|---|
| Output | 3840x2160, HEVC Main10, yuv420p10le, BT.2020 / PQ |
| HDR reference | 3840x1600, HEVC, `hevc_cuvid` |
| SDR Open Matte | 3840x2160, H.264, `h264_cuvid` |
| Overlap rect | `[0, 280, 3840, 1880]` (1600 rows, 1:1 mapping, no resampling) |
| Extension bands | rows `0..280` and `1880..2160` — 26% of frame, **no HDR ground truth** |
| Frame offset (BR2049 t2714) | OM = HDR + 1167 |
| Fit scale | 0.5 → 1920x1080 fit grid |
| Surface ring | 4 GPU slots |
| Video path | NVDEC → `AV_PIX_FMT_CUDA` → V5 render → P010 → `hevc_nvenc`; H2D = 0, D2H = 0 |
| Production color model | `NativeV4GpuShotFitter.fit_shot → v1.predict → v4.apply_intensity_conditioned` (`intensity_joint`) |
| Not in production | HPPM, HCAPM, global matrix, CDF, residual layer — all diagnostic-only, all NO-PASS |

### Measured baseline (500-frame test)

| Phase | Wall time | Per frame |
|---|---|---|
| fit | 22.088 s | 44.2 ms |
| render | 62.164 s | 124.3 ms |
| total | 84.940 s | 169.9 ms (5.89 FPS e2e) |

Throughput target: **≤ 41.6 ms/frame end-to-end** (24 fps ⇒ ≤ 1x material length for a 2 h feature ≈ 172,500 frames).

---

## 1. Ground rules

1. **Do not modify the frozen baseline** (`fit_shot`, `v1.predict`, `v4.apply_intensity_conditioned`, gain field, spatial field, renderer, NVDEC/CUDA/NVENC transport, source media) except where a task below explicitly authorizes it. Performance tasks 1, 3, 8 must be **bit-behaviour-preserving within a declared tolerance** — see each task.
2. **Every task is a separate branch and a separate PR.** Do not bundle a performance fix with a quality change; they need independent A/B evidence.
3. **Predeclare acceptance criteria before running the measurement.** Write them into the PR description first, then run. Do not tune the threshold to the result.
4. **Numerical-regression guard for all performance work:** render the same 20-frame interval before and after, compare output P010 planes. Declare and justify a tolerance (a 1-code-value 10-bit difference is acceptable; a systematic shift is not). Record max abs diff and the fraction of differing pixels.
5. **Never commit source-derived video frames or full-resolution PNGs** to git. JSON reports and small diagnostic plots only.
6. **Working tree must be clean before starting.** See task 0.

---

## Task 0 — Repository hygiene (do this first, blocks everything else)

**Why:** Three diagnostic artifacts are local-only, and two production files carry uncommitted local modifications. Any measurement taken on uncommitted code is not reproducible and cannot be attached to a commit SHA, which breaks the project's own protocol discipline.

**Steps:**
1. Inspect the uncommitted modifications:
   ```
   git diff -- tools/openmatte_hdr/fast_chroma_field_intensity.py tools/openmatte_hdr/fastcore.py
   ```
2. Decide per file: commit (if intentional) or revert (if scratch). Do not leave them dirty. If committing, the message must state what changed and why.
3. Commit the three diagnostic artifacts — they are evidence:
   - `fit_scaling_100_250_500.json`
   - `fit_scaling_500f_profile.json`
   - `hcapm_full_overlap_zonal.json`
   Place them next to the existing benchmark evidence under `benchmarks/v05_gpu_native_segmented_test300/` unless a better-matching directory already exists.
4. Confirm `git status --porcelain` is empty.

**Acceptance:** clean tree; all three artifacts tracked; the two production files either committed with a rationale or reverted.

**Model:** Sonnet-class, low reasoning effort. Mechanical.

---

## Task 1 — Eliminate the `yuv_to_working_rgb` hotspot

**Priority: highest. This is the single largest GPU cost in the pipeline.**

**Evidence:** Nsight (`nsight_render20.json`, `render20.nsys-rep`, commit `d0ef113`): total GPU kernel time 2.926 s for 20 frames = 146.3 ms/frame. `yuv_to_working_rgb` accounts for **41.67%** ≈ **61 ms/frame**.

**Why this is a bug, not a tuning issue:** the kernel performs a YUV→RGB matrix plus a transfer-function decode. That is roughly a dozen FMAs per pixel, one global read, one global write. At 3840x2160 it should cost **0.3–1 ms**. 61 ms is 60–200x too slow.

**Primary hypothesis: FP64 arithmetic in the kernel.** GA102 executes FP64 at 1/64 the FP32 rate. A single `pow(x, 2.4)` instead of `powf(x, 2.4f)`, or an unsuffixed literal such as `1.0/2.4`, promotes the expression to double and costs exactly this order of magnitude.

**Steps:**
1. Confirm or reject the FP64 hypothesis before changing anything:
   ```
   ncu --kernel-name yuv_to_working_rgb \
       --metrics smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
   smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,\
   smsp__sass_thread_inst_executed_op_dadd_pred_on.sum \
       <render command, 20 frames>
   ```
   Non-zero `d*` counters confirm it. Alternative: build with `nvcc -Xptxas -v` and grep the SASS for `DADD` / `DMUL` / `DFMA`.
2. If FP64 is confirmed: add `f` suffixes to every float literal in the kernel and switch to the single-precision intrinsics (`powf`, `fmaf`, `__saturatef`). Re-measure.
3. Regardless of step 2's outcome, **remove the transcendental entirely**: replace the PQ/gamma evaluation with the sqrt-domain PQ LUT already validated in `P2.17_PQ_OETF_SQRT_LUT_PRODUCTION_INTEGRATION_REPORT.md`. Load the table into `__constant__` or shared memory.
4. If the kernel is still slow after 2 and 3, the secondary suspect is uncoalesced access on the P010 UV plane. Check with:
   ```
   ncu --kernel-name yuv_to_working_rgb \
       --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
   l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
   ```
   Sectors-per-request materially above 4 for a 32-bit access pattern indicates poor coalescing; restructure to vectorized (`ushort2` / `uint4`) loads.

**Acceptance:**
- `yuv_to_working_rgb` ≤ **3 ms/frame** at 3840x2160 (a ≥ 20x improvement).
- Numerical-regression guard from ground rule 4 passes on the same 20-frame interval.
- Nsight re-profile committed as a new artifact showing the new kernel distribution.

**Model:** Opus-class, **high** reasoning effort. Requires reading SASS/profiler output, reasoning about precision promotion rules and memory access patterns, and preserving colour-transform correctness while changing the numerical path. Do not delegate this to a small model — a wrong `f` suffix or a mis-indexed LUT silently shifts every pixel.

---

## Task 2 — Make the fit O(1) in VRAM

**Priority: high. This is a hard scalability failure, not a performance issue.**

**Evidence:** `fit_scaling_100_250_500.json`:

| Frames | Peak VRAM |
|---|---|
| 100 | 4.13 GB |
| 250 | 4.86 GB |
| 500 | 6.09 GB |

That is ≈ **4.9 MB of resident VRAM per frame**. Extrapolating on 10 GB: `4.13 + 0.0049 × (n − 100) = 9.5` ⇒ OOM at **≈ 1200 frames ≈ 50 s of footage**. Any longer take fails, and multi-shot batching is impossible.

**Why it is wrong:** a per-shot global fit needs only the normal equations — `XᵀX` (≈ 60x60) and `XᵀY` (≈ 60x3) — which is a few kilobytes **independent of shot length**. Linear VRAM growth means frames or per-frame arrays are being kept alive for the duration of the fit.

**Steps:**
1. Locate the allocation that scales with frame count in the `NativeV4GpuShotFitter.fit_shot` path. Instrument with `cudaMemGetInfo` (or CuPy's `mempool.used_bytes()`) at frame granularity to identify what is retained.
2. Restructure to a **streaming accumulator**: for each frame, compute the sample contributions, accumulate into fixed-size `XᵀX` / `XᵀY` buffers, then release the frame surface back to the ring immediately.
3. Verify the ring is actually recycling: `acquired`, `released` and `in_flight` counters must behave as in the current 500-frame reliability run (`acquired=500`, `released=500`, `in_flight=0`).
4. Re-run the 100 / 250 / 500 scaling benchmark and add a **2000-frame** point, which currently cannot run.

**Acceptance:**
- Peak VRAM **flat within ±10%** across 100 / 250 / 500 / 2000 frames.
- 2000-frame fit completes without OOM.
- Fitted coefficients match the pre-change fit within a declared tolerance on the same interval (this is a refactor, not a model change — the output must be equivalent).

**Model:** Opus-class, **high** reasoning effort. Lifetime and ownership analysis across a Python/native boundary with a surface ring is exactly where a weaker model introduces use-after-release or leaks. The correctness bar is "identical coefficients", which requires care.

---

## Task 3 — Multi-scale Gaussian

**Evidence:** Nsight: `correlate*` = 1.166 s / 20 frames = **58 ms/frame**, **39.86%** of GPU time, **222 calls** (≈ 11.1 per frame — consistent with a separable 2-axis filter over ~5–6 planes). With `sigma = 16` and `truncate = 4.0` the kernel radius is 64, i.e. 129 taps per axis, running through a generic, non-tiled ndimage path.

**Why it is safe to change:** the field is low-frequency **by construction** — this is the founding assumption of the "bounded low-frequency chroma field" design. Filtering it at full 4K resolution is paying 64–256x for information that is not there.

**Steps:**
1. Replace each full-resolution `sigma = 16` blur with: **downsample 16x (area) → Gaussian with the correspondingly small sigma → bilinear upsample**.
2. Handle borders to match the current `mode="mirror"` behaviour; border handling is where this refactor most often diverges.
3. Quantify the approximation: on a representative frame, compute max and RMS difference between the full-resolution blur and the multi-scale blur, in the same units the field is consumed in.
4. Confirm the low-pass chroma residual metric is unchanged within tolerance — this pipeline's own headline metric depends on the blur.

**Acceptance:**
- Blur cost ≤ **2 ms/frame** total (from 58 ms).
- Low-pass chroma residual and seam chroma step change by **< 1%** relative to baseline on the reference interval.
- Approximation error reported explicitly in the PR.

**Model:** Sonnet-class with **medium** reasoning effort is sufficient if the agent is disciplined about border handling and reports the approximation error; escalate to Opus-class if the residual metric moves more than 1% and the cause is not obvious.

---

## Task 4 — Stratified sampling in the fit

**This is the only task that improves throughput and quality simultaneously. If only one task is done, do this one.**

**Evidence (throughput):** `fit_seconds = 0.968 + 0.03998 × frames`, R² = 0.99992. Marginal cost 39.09 ms/frame (100→250) and 40.43 ms/frame (250→500). For 172,500 frames that is **1.92 h of fitting alone**; with ~2000 shots the 0.968 s fixed overhead adds a further **0.54 h** of pure setup.

**Evidence (quality)** — `MODEL_DISCRIMINATION_REPORT.md`, overlap distribution by HDR reference luminance:

| Band | Pixels | Share | Luma MAE | Signed RGB residual (R,G,B) nits |
|---|---|---|---|---|
| 0–5 nits | 963,101 | **62.7%** | 3.15 | +3.9 / +2.7 / +1.4 |
| 50–100 | 117,517 | 7.7% | 30.06 | −32.5 / −23.8 / −8.0 |
| 100–250 | 17,499 | **1.1%** | **48.86** | **−61.5 / −44.1 / −10.6** |
| > 250 | **0** | 0% | — | — |

An unweighted fit is dominated by near-black. Highlights are 1.1% of samples and are drowned out, producing a **monotonic, single-signed** error that reaches −61 nits in R. Above 250 nits the fit has **no samples at all**.

**Steps:**
1. Replace the current dense fit input (1920x1080 x every frame ≈ 1.04e9 samples for a 500-frame shot) with **stratified sampling of ≈ 200,000 pairs per shot**: about 16 frames spread across the shot, ≈ 12,500 samples per frame.
2. Stratify jointly over **luminance bin x chroma bin x hue sector x row band**, with a **cap per cell**. The row-band axis is required — task 6 depends on spatial coverage.
3. Report per-cell sample counts in the fit JSON. Cells below a minimum count must be flagged, and the fit must shrink toward identity where support is thin.
4. Keep the accumulation on GPU (it is a tiny reduction), but **solve on CPU in float64** via Cholesky with ridge (`scipy.linalg.cho_factor` / `cho_solve`). Float64 conditioning matters here and the solve is microseconds.
5. Hoist the 0.968 s fixed overhead out of the per-shot loop (surface allocation, context setup) so it is paid once per run, not once per shot.
6. Re-measure fit wall time and re-measure the per-luminance-band residual table above.

**Acceptance:**
- Fit wall time for a 500-frame shot ≤ **1.5 s** (from 20.987 s).
- Signed RGB residual in the 100–250 nit band improves by **≥ 50%** in magnitude (from −61.5 / −44.1 / −10.6).
- No band regresses by more than a predeclared tolerance.
- Per-cell sample counts present in the output JSON.

**Model:** Opus-class, **high** reasoning effort. Stratification design, weighting, conditioning and the shrinkage rule for thin cells are statistical decisions with quality consequences; a weaker model will implement uniform subsampling and lose the entire quality benefit.

---

## Task 5 — Test x-dependence of the seam gradient

**Evidence and motivation:** `hcapm_full_overlap_zonal.json` measured 16 bands of 100 px across the full 1600 px overlap (HDR `[65359,65371)`, OM `[66526,66538)`), and found:
- **Chroma:** median Δlog C deepens monotonically from ≈ −0.0065 at the top edge to ≈ −0.022 at the bottom, across the **entire** overlap.
- **Hue:** median ΔH is slightly positive at 100–400 px, **crosses zero at ≈ 800 px** (the geometric centre of the 1600 px overlap), then reaches ≈ −0.75° and stabilises — i.e. **antisymmetric about the frame centre**.

That measurement aggregates over full rows, which **averages out any horizontal variation**. Task 6 assumes a 1-D model in y; this task validates that assumption before it is committed to.

**Steps:**
1. Re-run the same 16-band analysis, additionally split into **6 segments along x** (640 px each).
2. Report Δlog C and circular median ΔH per (band, x-segment) cell, with sample counts.
3. Verdict rule, predeclared: if the spread across x-segments within a band is **below the bootstrap confidence interval** of the per-band median, the effect is 1-D in y and task 6 proceeds as specified. Otherwise task 6 must be widened to a 2-D grid.
4. Separately, determine **which master carries the gradient**: compute mean chroma per row band on the **HDR reference alone**, with no reference to the OM. A vertical trend of comparable magnitude localises the gradient to the HDR grade; its absence points to the OM (H.264 WEB-DL, different master).

**Acceptance:** a JSON report with a clear `1D_IN_Y` / `REQUIRES_2D` verdict and the HDR-only row-trend result. **No renderer, no NVENC, no video output.** Diagnostic only.

**Model:** Sonnet-class, **medium** reasoning effort. This is a measurement harness modelled closely on an existing one. Escalate for the bootstrap/CI logic if the agent is unsure how to construct it.

---

## Task 6 — Spatial layer L2 as a 1-D field in y

**Blocked by task 5.**

**Why the previous attempts failed:** HPPM applied corrections in **32 px** edge bands; HCAPM used profiles centred at **10 / 50 / 90 px**. The measured effect spans **1600 px**. All seven experiments modelled 2–6% of the phenomenon's extent and applied the correction only within that window — capturing a fraction of the signal while adding estimation noise. Combined with a training domain (`C ≥ 0.10`) that did not match the application gate (full strength from `C ≥ 0.08`, partial from `0.03`), and a PASS rule requiring both ICtCp `I` and BT.2020 luminance to be exactly unchanged (impossible for any polar Ct/Cp correction, since `I` is not luminance), the failures were structural rather than model-capacity limits. **The zonal idea was correct; its extent and its gate were not.**

**Design, if task 5 returns `1D_IN_Y`:**
- `Δlog C(y)`: monotone, smooth. 8–16 control points over 2160 rows, or a degree-2/3 polynomial.
- `ΔH(y)`: antisymmetric about the frame centre, constrained to pass through zero there. Odd polynomial (cubic) or 8–16 control points with the zero enforced.
- **Total ≈ 16–32 parameters for the whole spatial layer** (versus the ~4096-node grid that a 2-D design would need).
- Fit on rows `280..1880` only. Extrapolate into rows `0..280` and `1880..2160` by smooth continuation **with shrinkage toward identity**, preserving C1 continuity so no new seam is created.
- Apply cost: one lookup into a 2160-entry table per pixel.

**Corrected gate and PASS rule — mandatory:**
1. Application gate floor **≥ training domain floor**. If training uses `C ≥ 0.10`, the gate must be `W = 0` below `0.10` with a smoothstep to `0.15`. Medium-chroma groups then cannot regress by construction.
2. Primary metrics: **ΔE_ITP** and **seam chroma step**. Raw hue in degrees is ill-conditioned near the achromatic axis — this is why earlier reports showed 21–25° p95 in medium-chroma groups against 2.7–4.9° in high-chroma ones. Use chroma-weighted circular error if a hue metric is needed.
3. **Holdout split by frames, not by pixels.**
4. PASS rule with a **declared indifference band** (e.g. a regression below 0.05° counts as unchanged) plus bootstrap confidence intervals over frames. A prior report rejected a configuration on a 0.00063° regression, which is float32 noise.
5. Luminance invariant: **ΔI = 0 exactly, plus a tolerance on BT.2020 Y**, justified against the 10-bit PQ code step. Requiring both exactly is unsatisfiable; the previously measured 0.0373 nits is below one code value across most of the range.

**Acceptance:** ΔE_ITP and seam chroma step improve on a frame-held-out split, no group regresses beyond the declared indifference band, RGB range safety holds (`nonfinite_count = 0`, `preclip_out_of_range_fraction = 0`), and manual HDR visual review is still recorded as required.

**Model:** Opus-class, **high** reasoning effort. This is the core quality change and it must not repeat the gate/domain mismatch that invalidated seven prior experiments.

---

## Task 7 — Fix the linearity gate statistic

**Evidence:** `fit_scaling_100_250_500.json` reports `not_confirmed_as_linear` because mean ms/frame deviated 10.94% against a 10% threshold. The gate tests the wrong quantity. With an intercept, mean ms/frame is `0.968/n + 0.03998`, which **decreases with n by construction**. The 10.94% figure is amortisation of fixed overhead, not non-linearity. The correct statistics both confirm linearity strongly: R² = 0.99992, and marginal cost is 39.09 vs 40.43 ms/frame between consecutive points — a 3.4% spread.

**Steps:** change the criterion to the spread of **marginal** cost between consecutive measurement points (or the residuals of the linear fit), keep the intercept explicit in the report, and re-issue the verdict. Do not silently change the recorded history — note that the earlier verdict was a false negative and why.

**Acceptance:** report states `linear` with the marginal-cost spread and R² as evidence, and documents the superseded verdict.

**Model:** Sonnet-class, **low** reasoning effort. Small, well-specified change.

---

## Task 8 — Fuse the per-pixel path and bake L1 into a 33³ LUT

**Do last.** Tasks 1 and 3 remove ~81% of current GPU time; this task addresses what remains (~27 ms/frame) and decouples quality from cost permanently.

**Core idea:** whatever per-shot colour-volume model is fitted (`intensity_joint`, MMR-1, richer), **bake it once per shot into a 33³ lattice** and apply it per pixel by tetrahedral interpolation. 33³ = 35,937 nodes against ~4.15e9 pixels for a 500-frame 4K shot — baking is ~115,000x cheaper than evaluating the model per pixel. Per-pixel cost then becomes **independent of model complexity**, which removes the quality-versus-speed tradeoff entirely.

**Steps:**
1. One fused kernel: unpack P010 → PQ EOTF via LUT → BT.2020 → tetrahedral lookup in the 33³ table → L2 slice (task 6) → PQ OETF via LUT → pack P010. One global read, one global write; everything else in registers. Grid-stride loop, 256 threads/block, `__restrict__` on all pointers.
2. **Do not use hardware texture filtering for the 3-D LUT.** The texture units interpolate with 9-bit fixed-point (1.8 format) weights, which is far too coarse for 10-bit PQ and will band in gradients — and the banding will look like a model error. Fetch in point mode and compute **tetrahedral** interpolation in fp32. Tetrahedral also beats trilinear at equal LUT size.
3. **Do not enable `--use_fast_math` globally**; it changes PQ accuracy and invalidates the numerical gates. Use explicit `__fmaf_rn` where needed and keep the LUTs.
4. LUT construction must handle **unsupported nodes**: build a per-node sample count from the fit data, solve the lattice with a Laplacian smoothness term so that zero-support nodes acquire values by diffusion from supported neighbours, and shrink toward identity where support is thin. This is the principled answer to colours that appear in the extension bands but not in the overlap. Constrain monotonicity along luma to prevent inversions in extrapolation.
5. Validate the colour path against **OpenColorIO** on synthetic ramps as an independent correctness reference for PQ, BT.2020 and tetrahedral interpolation. Cheap regression test for banding.

**Acceptance:**
- Fused per-pixel path ≤ **5 ms/frame** at 3840x2160.
- Output equivalent to the unfused path within a declared tolerance on the reference interval.
- No banding on synthetic ramps versus the OCIO reference.
- Nsight profile committed showing NVDEC/NVENC, not the SM kernels, as the remaining ceiling.

**Model:** Opus-class, **high** reasoning effort, and expect several iterations. Hand-written CUDA with a scattered-data lattice solve, extrapolation safety and a numerical-equivalence bar is the hardest item in this brief.

---

## Ordering and dependencies

```
Task 0  (hygiene) ──> everything

Performance track (independent of colour):
  Task 1  (yuv_to_working_rgb)  ─┐
  Task 3  (multi-scale Gaussian) ┼──> Task 8 (fuse + bake LUT)
  Task 2  (fit VRAM O(1))       ─┘

Quality track:
  Task 4  (stratified sampling) ──> Task 6 (L2 1-D field)
  Task 5  (x-dependence test)   ──> Task 6

Independent:
  Task 7  (linearity gate)
```

Tasks 1, 2, 3 and 7 are independent of any colour decision and can proceed in parallel with the quality track. Task 4 sits on both tracks and is the highest value-per-effort item in the brief.

## Projected outcome

| Metric | Now | After tasks 1–4, 8 |
|---|---|---|
| `yuv_to_working_rgb` | 61 ms/frame | ≤ 3 ms |
| Gaussian | 58 ms/frame | ≤ 2 ms |
| Render total | 124.3 ms/frame | ~5 ms |
| Fit (500-frame shot) | 20.987 s | ≤ 1.5 s |
| Peak VRAM | grows 4.9 MB/frame | flat |
| End-to-end | 5.89 FPS (≈ 4.1x material) | NVDEC/NVENC-bound, ~0.7–1.0x material |

---

## A caution about priorities

Convert the measured seam gradient into perceptual units before investing heavily in task 6. At a typical `C ≈ 0.1`, `Δlog C = −0.022` is a 2.2% chroma reduction (`ΔC ≈ 0.0022`) and `ΔH = 0.75°` is an arc of ≈ 0.0013 — together **of order 1–2 ΔE_ITP**. Phase 16 reported holdout ΔE_ITP macro means of **48.20–50.35**. On those numbers the row gradient is roughly **2–4% of total error**.

Compute this exactly on the project's own data. If the estimate holds, then a visible quality mismatch is **not** primarily caused by this gradient, and the dominant term is more likely the highlight luminance path (48.86 nits MAE and a −61.5 nit signed residual in the 100–250 nit band, with zero samples above 250 nits) — which task 4 addresses directly.

Task 6 is still worth doing: at 16–32 parameters it is cheap and it is now properly measured. But it should not be framed as the fix for the perceived problem until the highlight path is ruled out.

---

## Model and reasoning-level summary

| Task | Model class | Reasoning effort | Rationale |
|---|---|---|---|
| 0 — hygiene | Sonnet | low | mechanical git work |
| 1 — `yuv_to_working_rgb` | **Opus** | **high** | profiler/SASS reading, precision-promotion rules, correctness of colour path |
| 2 — fit VRAM | **Opus** | **high** | lifetime/ownership across Python↔native with a surface ring |
| 3 — Gaussian | Sonnet | medium | localised, but border handling is subtle |
| 4 — stratified sampling | **Opus** | **high** | statistical design with direct quality consequences |
| 5 — x-dependence test | Sonnet | medium | mirrors an existing harness |
| 6 — L2 1-D field | **Opus** | **high** | core quality change; must not repeat prior gate errors |
| 7 — linearity gate | Sonnet | low | small, fully specified |
| 8 — fuse + bake LUT | **Opus** | **high** | hardest item; hand-written CUDA plus lattice solve |

**General guidance:** anything that touches the numerical colour path, the CUDA kernels, or the statistical design of a fit should run on the strongest available model with extended thinking enabled. Tasks 0, 3, 5 and 7 are safe on a mid-tier model provided the acceptance criteria in this brief are enforced verbatim. In all cases the agent must have a working local build and the RTX 3080 available — none of the acceptance thresholds here can be verified without running on the target GPU.

# Where to take this next

The bet: **NsNet2's suppression mask is a global speech-presence gate, not a per-bin SNR
estimator — and that is a characterisation worth publishing and worth acting on.**

Evidence already in hand (`REVIEW.md`, reproducible from `diagnostics/` and
`DeepShap/validate_attributions.py`): the input-frequency × output-frequency coupling matrix
is rank one on real speech (top-1 singular value 0.946–0.992, participation ratio ~1.07,
diagonal enrichment ~1.2) while an oracle Wiener gain measured through the identical
pipeline comes out strongly local (enrichment 15.4, participation ratio ~116). The apparatus
detects locality; this model has none.

That object is exactly computable — 257 backward passes on one forward graph, no baselines,
no completeness axiom, no Captum hooks, no need to decompose the GRUs. It is therefore
immune to every defect that sank the original pipeline, which is why it leads the plan
rather than following a long methods bake-off.

Assumed resources: one student, 3–6 months, one GPU, Valentini VCTK+DEMAND.

---

## Phase 0 — Unbreak and freeze the substrate (week 1, hard cap)

Nothing computed after this week should need recomputing.

- Add `tests/`. There is none. Minimum: the STFT contract (`win_length == 512`,
  `hop_length == 256`, `load_state_dict(strict=True)` succeeds), the
  `sigmoid(mask_logits(log_power(x))) == _forward(x)` identity, output length preservation,
  and a `main.py` smoke test with default arguments.
- **Pin the STFT config with that test, not a comment.** It flip-flopped twice during this
  review. 512/256 is correct: the checkpoint's `preproc.window` is exactly
  `torch.hann_window(512)`, `win_length=320` makes `load_state_dict` raise, and measured
  SI-SDR gain across five noise types at 0–15 dB puts 512/256 at the top (+3.6 dB mean)
  against +0.9 dB for 320/160. The 20 ms/10 ms figure in the DNS README describes the
  public **161-bin** NSNet2, a different checkpoint from the 257-bin file committed here.
- **Resolve three hop lengths that currently coexist**: `config/parameters.py` says 128,
  `NsNet2_model.py` says 256, `data_utils.compute_time_bands` defaults to 512. The legacy
  path only runs because two of the wrong values agree with each other.
- **Fix `main.py`'s own import contract.** `python -m DeepShap.main` still fails with
  `ModuleNotFoundError: No module named 'utils'` — line 8 appends the repo root where it
  needs `DeepShap/`. It works only via `python DeepShap/main.py`.
- **Delete rather than refactor**: `models/frequency_time_feature_model.py` (wrong
  explanandum), `utils/attributions_utils.py` (the `.mean()` collapse),
  `convert_attr_map_to_mel_scale`, `deep_shap_test.py` (cannot run at all). Leaving them
  invites reuse. Note `activation_visualise.py` and `deep_shap_test.py` each hand-reimplement
  `mask_logits` with functional nonlinearities, so the fixed model does not protect them.
- Replace the six frequency bands (3/4/8/16/64/160 FFT bins) with a 32-band ERB set **and**
  a 32-band equal-bin-count set, both selectable and both recorded in every output.
- Download only Valentini `clean_testset_wav` + `noisy_testset_wav` + `logfiles` (310 MB,
  CC BY 4.0, 824 utterances, 5 unseen noises × 4 SNRs). Resample to 16 kHz, build an index,
  cache the oracle IBM (`|clean| > |noise|`) and clean-reference VAD — all three are free and
  needed by later phases. Do **not** pull DNS Challenge bulk data.
- **Ask the supervisor, in writing, where `nsnet2_baseline.bin` came from.** It is a 257-bin
  model; Microsoft released a 161-bin one. Its window buffer proves it was re-saved from this
  repo's own `nn.Module`, so it carries no provenance — and every result rests on it.

**Gate:** `pytest` green; `python -m DeepShap.main` runs on 10 real files with default args.

## Phase 1 — Bank the flagship at corpus scale (weeks 2–4)

- Compute `R[f_in, f_out]` over 300 Valentini utterances, 8 output frames each, stratified
  speech/pause by the clean-reference VAD. ~13 s per output frame on 4 CPUs; trivial on a GPU.
- Report top-1 singular-value share, participation ratio and diagonal enrichment with
  bootstrap CIs, stratified across all 20 conditions.
- **Run the oracle-Wiener control every single time.** If it stops showing locality, the
  metric is broken and the finding is void.
- Robustness variants: seeds × SNRs; the **signed** rather than absolute coupling (rules out
  sign-cancellation); and lag-0 versus lag-4, where the instantaneous response appears to
  carry more structure that collapses within ~4 frames — a possible two-timescale
  decomposition nobody has looked at.
- Cross-check with FreqRISE (Bernoulli p=0.5 masks on a 25×25 grid, bilinear upsample,
  N≈3000, weighted by the mask **logit**), which owes nothing to gradients or hooks. The
  Jacobian is a gradient object and ~39% of the plane is saturated, so the claim needs one
  confirmation from a non-gradient method.

**Gate:** corpus top-1 SV > 0.80 with a CI; diagonal enrichment < 2.0 in every condition;
oracle Wiener > 10 in the same pipeline; FreqRISE agreeing with the Jacobian's input-frequency
marginal at Spearman ≥ 0.6.

## Phase 2 — Make it causal (weeks 4–7)

Rank-one is correlational. Attack the same object from two sides.

- **Band removal:** `R` predicts a sensitivity for each `(f_in, f_out)`. Remove input band
  `f_in` — ROAD neighbour imputation, never zero, since zero means `log(1e-8)` = −18.4, off
  manifold and deep in saturation — and measure the true change in the `f_out` logit.
  Report Spearman between predicted and measured.
- **Directional ablation, not neuron ablation.** Train logistic probes on the 400-d GRU2
  state for speech presence, noise class and instantaneous SNR (ground truth is free from the
  clean reference and the mixing recipe), then patch out the speech-presence *direction*
  during streaming. Controls: 50 random unit-norm directions at matched norm, plus top-k
  single-unit ablation, which prior probing suggests is near-null. The prediction is that the
  direction works where the neurons do not.
- **Test the product-relevant prediction.** Rank-one implies a narrowband interferer in the
  dominant input band should gate the *whole* spectrum. Preliminary evidence already agrees:
  a burst 20 dB above the speech barely changes the mask in its own band. Measure this
  properly against DNSMOS SIG drop. A confirmed prediction is a mechanism claim; a refuted
  one is equally reportable.

**Gate:** band-removal Spearman > 0.6; directional ablation cuts the speech/pause gain
contrast by ≥ 40% at ≥ 3σ over the matched-norm random control.

## Phase 3 — The dynamics report card (weeks 6–9, runs alongside)

Pure forward-pass work that cannot fail to produce numbers, and the most directly
product-relevant output. Rewrite `activation_visualise.py` into a measurement rig — its SNR
axis is currently mislabelled by 5–15 dB (mean-absolute instead of RMS, and it ignores the
burst duty cycle).

- **Effective memory horizon:** truncate the context to L frames, report the L at which mask
  divergence falls below 1%, in **milliseconds**, per noise type and SNR.
- Report per-lag gradient mass, single-frame occlusion and cumulative truncation on the same
  axes, and treat their disagreement as the finding — single-frame effects appear to die
  within ~256 ms while cumulative truncation still moves the logit at 2 s. That collective
  long tail is Ancona's sensitivity-n phenomenon on a real audio model, and it is the honest
  answer to why single-bin attributions look unconvincing.
- Attack/release per ERB band, hysteresis loop, musical-noise prevalence. A gate-reopen time
  in milliseconds is exactly the spec-sheet number a headset engineer can act on.

## Phase 4 — Method admissibility benchmark (weeks 9–13, hard cap)

Close out the DeepSHAP work honestly as a chapter, not as the project.

- Open with the saturation histogram (39.3% of logits at |z| > 4.6): a one-forward-pass
  death certificate for the sigmoid target.
- Methods: saliency, input × gradient, integrated gradients, gradient SHAP with 32 *distinct*
  real reference frames, DeepLIFT, DeepLiftShap, SmoothGrad-Squared and VarGrad (not mean
  SmoothGrad — ROAR shows it is worse than the base method at 15× the cost), feature ablation,
  occlusion, KernelSHAP over grouped patches, FreqRISE. **Random and energy-detector controls
  on every row**, plus the legacy pipeline as a calibration row that must come out
  weight-independent.
- Extend the burst probe from n=1 to ≥ 200 randomised (frequency, time, SNR) placements.
  Report Arras relevance rank accuracy and pointing-game hit rate with CIs alongside
  enrichment. Keep the displaced probe; the co-located one is circular.
- Sanity checks as an **admissibility gate**: cascading and independent parameter
  randomisation over `fc4 … fc1`, both directions, reporting abs-Spearman, no-abs-Spearman,
  SSIM and HOG separately. A method that fails is not scored on anything else.
- Frame every negative verdict as "inadmissible for NsNet2 mask-logit explanation", never as
  a universal verdict on the method.

**Gate:** if no method beats the energy detector, that is the headline and the chapter closes
early. On present evidence the margin is about 2.8×, so this is a real test.

## Phase 5 — Does any of it predict quality? (weeks 12–18)

This is the move that made the closest prior work (Sivasankaran et al., Interspeech 2021)
publishable, and the only thing that turns an explanation into an instrument.

- Implement their **speech relevance score η**: of the input TF bins whose |relevance|
  exceeds the T-th percentile within a frame, the fraction that are speech-dominated under
  the IBM, over speech frames only, with T pre-registered at 99.9/99.0/98.0 and all three
  reported. Valentini's parallel pairs make the IBM free. It is precision-only by
  construction — do not invent a recall.
- Correlate η, the Phase 1 top-1 SV share and the Phase 3 memory horizon against ΔPESQ,
  ΔESTOI, ΔSI-SDR and DNSMOS SIG/BAK/OVRL. Report utterance-level and condition-level
  separately.
- **Partial out SNR.** Both η and quality track input SNR, so the raw correlation is likely
  vacuous. `logfiles.zip` gives the per-utterance SNR.
- **Attribute DNSMOS SIG and BAK separately** — nobody has, and for a mask suppressor the
  divergence between them (noise removed vs speech damaged) is the whole story.
- Licence note for a commercial context: the P.862 core is evaluation-only and NISQA weights
  are CC BY-NC-SA. Use DNSMOS (CC BY 4.0) or Distill-MOS (MIT) for anything informing a
  product decision.

**Gate:** |partial Spearman| ≥ 0.3 at utterance level with a CI excluding zero, or ≥ 0.7 at
condition level, *after* partialling out SNR. Otherwise report the null with a power analysis
and close the phase — no subgroup hunting.

## Phase 6 — Failure-mode atlas and the recommendation (weeks 18–24)

- Rank all 824 utterances by DNSMOS SIG drop and ESTOI drop, cluster the worst decile by
  noise type, SNR, speaker F0 and phone class, and name each mode with a measured prevalence
  and CI ("over-suppression of unvoiced fricatives below 8 dB SNR in babble: X% of /s/ tokens
  lose > 6 dB").
- Build a **reference-free early-warning signal** from the Phase 2 probes and Phase 3
  dynamics statistics, tested on held-out speakers and unseen noises. This is deployable as a
  runtime confidence signal on a headset, not just a report figure.
- Test **one** mitigation end to end. Cheapest first: a per-band release-smoothing rule
  derived from the Phase 3 numbers, which needs no retraining.
- Write the strategic section, and be willing to conclude against the project's own premise:
  for a hearables product line, an interpretable-by-design denoiser may beat explaining a
  black box. Recent work gets within 0.02 HASPI/HASQI of a 2.3M-parameter baseline using
  ~24k parameters over differentiable IIR biquads.

---

## What I would not do

- **Do not chase DeepSHAP.** Even with the nonlinearities registered correctly, the
  convergence delta stays high because the GRU gates are undecomposable by the rescale rule.
  The closest prior work hit the same wall in the same library and simply declared it. Use
  integrated gradients where a gradient method is needed, and FreqRISE where it is not.
- **Do not build listenable explanations early.** For a denoiser, "explanation audio" is
  uncomfortably close to the enhanced audio, which smuggles the pass-through problem back in
  audible form. It needs a redefined objective first — a research question, not an
  engineering one.
- **Do not use the co-located probe.** It scores a method on returning mass to the cell it
  was asked about, which penalises exactly the non-local behaviour this model actually has.
- **Do not report a single-file number again.** Every claim gets an n, a bootstrap CI, and a
  decision rule fixed before the run.

## Open questions worth an early answer

1. What *is* the checkpoint? 257 bins, no provenance, and everything rests on it.
2. Is rank-one a property of NsNet2, of GRU denoisers, or of masking denoisers generally?
   A second model — the public 161-bin NSNet2, or DeepFilterNet — settles it cheaply and
   multiplies the result's value.
3. Does the gate hold at SNR extremes, or does the code get richer at low SNR? The
   stratification is free from the Valentini metadata.
4. What does "removal" mean for a TF representation with phase? If the operator spread
   exceeds the method spread, that instability is itself the publishable finding — but it
   also means deletion AUC cannot be used to select methods.
5. Publication or product input? Phase 4 serves the first, Phase 6 the second. Worth asking
   before week 9.

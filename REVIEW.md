# Review: XAI for speech enhancement (NsNet2 + DeepLiftShap)

Reviewed at commit `02f649e`. Every claim below was reproduced by running the code; the
supporting script is `diagnostics/diagnose_attribution_pipeline.py`, which needs no dataset.

## Summary

The pipeline could not have produced informative attribution maps, for reasons that are
structural rather than a matter of tuning. Three defects compound:

1. **It was not running DeepLIFT.** Captum applies the DeepLIFT rescale rule only to
   nonlinearities that are registered `nn.Module` instances. `NsNet2_model.py` used
   `nn.functional.relu` and `torch.sigmoid`, so attribution silently degraded to
   Gradient × Input. Proof: two numerically identical toy networks differing only in how
   ReLU is spelled — the functional one returns attributions *exactly* equal to
   `grad × (input − baseline)` with convergence delta 4.15; the modular one is exact,
   delta 0.00. On the real pipeline the mean convergence delta was **89.3 against a mean
   explained quantity of 93.6 dB**: the attributions accounted for about 5% of what they
   claimed to decompose.

2. **The saved map contained no input localisation, by construction.**
   `attributions_utils.py:48` did `attr_map[f, t] = attrs[idx].mean()`, collapsing a
   whole-waveform attribution vector to a single scalar. Both axes of the resulting map
   index the *output* cell being explained; nothing anywhere recorded *where in the input*
   the evidence lay. Reproduced on a probe signal with a 3 kHz burst at a known time: the
   map is near-flat (coefficient of variation 0.19) and the burst is invisible.

3. **The explanandum was the wrong quantity.** The wrapper explained the *enhanced
   output's* band energy. Since `enhanced = noisy × mask`, that target is dominated by the
   trivial pass-through path rather than by the model's denoising decision.

Contributing factors: attribution was taken with respect to the raw waveform, where
attributions oscillate in sign at audio rate and a sign-preserving mean cancels; the
baselines were ten identical copies of one tensor, a zero-variance "distribution" costing
~10× the compute of a single baseline for no extra information; a zero waveform maps to
`log(1e-8) = −18.4` in the log-power domain, far off-manifold and deeply saturated; and the
six frequency bands map to 3, 4, 8, 16, 64 and 160 FFT bins respectively, so the 3–8 kHz
band averages 160 bins into one number, destroying exactly where fricatives and most
broadband noise live.

Even if the attributions had been sound, the plot would have obscured them: the attribution
panel used the diverging `seismic` colormap with no zero-centred norm, and its colorbar was
built from a norm-less `ScalarMappable`, so the scale shown was decorative and a single
outlier cell would flatten everything else to mid-grey.

## One hypothesis raised and rejected

The STFT analysis window looked like a fourth root cause — `NsNet2_model.py` takes
torchaudio's defaults (32 ms window, 16 ms hop) where the DNS-Challenge NSNet2 baseline is
usually described as 20 ms / 10 ms. A mask-contrast proxy on a synthetic stationary stimulus
appeared to confirm it. **It is wrong.** Measured properly — SI-SDR gain on real speech
across five noise types at 0–15 dB SNR — the repo's 512/256 is at the top of the sweep
(+3.6 dB mean) and 320/160 scores +0.9 dB, losing up to 7.7 dB on some conditions. The
analysis window was never broken. Recorded here so nobody "fixes" it.

## Other defects found

| Where | Problem |
| --- | --- |
| `main.py` | `--noise_type` defaulted to `"not added"`, not in its `choices`. argparse does not validate defaults, so the default invocation reached `prepare_deepshap_input_and_baseline` and raised `ValueError`. |
| `main.py` | The 11 MB checkpoint was re-read from disk for every (file, division) pair. |
| `__init__.py` | Relative import beyond the top-level package; raised on any import. |
| `deep_shap_test.py` | Cannot run: `np.ndarray` has no `.abs()`, `.sum(dim=)` is invalid, it allocates a 2.0 GB array and would issue ~16k `DeepLiftShap` calls each saving a PNG. |
| `attributions_utils.py` | `convert_attr_map_to_mel_scale` builds a `[257 × n_audio_samples]` array — 164 MB for a 5 s file — then mel-interpolates a piecewise-constant 6-band map, inventing resolution that does not exist. |
| `model_utils.py` | Weights path resolved against the caller's working directory. |
| `data_utils.py` | `add_sinusoidal_noise` / `add_impulsive_noise` are unseeded; the injected SNR is never recorded. |
| `requirements.txt` | Unpinned, lists stdlib `argparse`, omits `scipy`, `dill`, `torchlens`, `Pillow`. `torchaudio >= 2.9` also needs `torchcodec` to load audio at all. |
| repo | No `.gitignore`; no tests; no metrics; no saved results; no seeds. |

## What was missing scientifically

No faithfulness metric, no sanity check, no control, and no comparison method — so "the
results were not promising" was not a measurable statement. Specifically absent: a
random-attribution control and an "attribution = input energy" control; Adebayo model- and
data-randomisation tests; deletion/insertion or infidelity/sensitivity (both are in Captum);
any method to compare against; any ground truth; any aggregation across a corpus; and any
measurement of whether the model was even doing a good job on the file being explained.

The one strong instinct in the original work is the injected-noise probe. Injecting a tone
at a known time and frequency *creates* a ground truth, and that is the seed of a real
benchmark — see `DeepShap/validate_attributions.py`.

## Reproduction after fixing

`DeepShap/validate_attributions.py` injects a narrowband burst at a known time and
frequency, so the right answer is known by construction, and scores every method on how
much attribution mass lands inside that region. `enrichment` is the mass fraction divided
by the region's area fraction, so **1.0 is chance**.

On a 3.4 s speech clip with a 300 Hz-wide burst at 3 kHz over 1.0–1.6 s (region = 1.53% of
the time-frequency plane). The model does suppress the burst — mean mask 0.24 inside the
region against 0.86 overall — so the question is well posed.

| pipeline | method | enrichment | mass in region | ‖convergence delta‖ |
| --- | --- | --- | --- | --- |
| original | DeepLiftShap on output band energy | **1.04×** | 1.59% | 89.3 |
| fixed | integrated gradients | **7.16×** | 10.99% | **0.14** |
| fixed | DeepLIFT | 6.90× | 10.59% | 2.72 |
| fixed | gradient SHAP | 6.95× | 10.65% | 4.42 |
| fixed | saliency | 4.14× | 6.35% | — |
| fixed | input × gradient | 1.64× | 2.51% | — |

The original pipeline scores **at chance**, and its map has a coefficient of variation of
0.064 — it is a constant plane. This is the quantitative form of "the results were not
super promising": the maps carried no information, so there was nothing to interpret.

Two further results are worth noting.

**Integrated gradients is the method to use here, not DeepSHAP.** Its convergence delta is
an order of magnitude smaller than DeepLIFT's even after the nonlinearities are registered
properly, because the two GRU layers remain undecomposable by the rescale rule. That is the
honest verdict on DeepLIFT/DeepSHAP for a recurrent masking model, and it is measured rather
than argued.

**All methods agree strongly with a brute-force occlusion sweep** (Spearman ρ = 0.81–0.90
over the full plane), which is assumption-free and therefore the best available faithfulness
reference. That agreement, more than the enrichment number, is what says the explanations
are now tracking the model.

A caveat on the enrichment metric: it scores localisation against *where the noise was
injected*, which is not necessarily *where the model looked*. The occlusion reference itself
concentrates less mass in the injected region (2.0x) than the gradient methods do, though it
was computed on 16x8-bin patches, which dilutes a region only 19 bins tall — how much of that
gap is patch granularity and how much is real is still open. Separately,
probing shows the mask decision at 3 kHz draws heavily on 125–650 Hz speech-band energy —
i.e. a global, VAD-like speech-presence decision rather than a per-bin one. That is a real
property of NsNet2, not a failure of the metric, and it is the kind of finding the original
pipeline was structurally incapable of producing.

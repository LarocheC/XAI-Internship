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
frequency, then explains the mask logit at a *different* region and asks how much
attribution mass lands on the burst. `enrichment` is mass fraction over area fraction, so
**1.0 is chance**, and two controls are reported on every row: a random ranking (null,
must come out at 1.0) and an energy detector that ranks bins by their own log-power.

A first version of this benchmark was circular — it explained the burst region *itself*,
so it scored a method on how much mass it returned to the cell it had been asked about.
That rewards diagonal attribution regardless of what the model does, and it inflated every
number; the ground-truth region was also twice as wide as the actual burst. Both are fixed;
`--probe colocated` still reproduces the old behaviour but prints a warning. The numbers
below are from the non-circular displaced probe.

3.4 s of real speech, 300 Hz burst at 3 kHz over 1.0–1.6 s, mask logit explained at 800 Hz.
Ground-truth region = 0.73% of the time-frequency plane.

| method | burst 0 dB | −10 dB | −20 dB |
| --- | --- | --- | --- |
| DeepLiftShap | 4.11× | 4.95× | **6.33×** |
| DeepLIFT | 4.13× | 4.95× | 6.32× |
| integrated gradients | 4.11× | 4.56× | 5.27× |
| gradient SHAP | 4.21× | 4.63× | 5.22× |
| input × gradient | 0.87× | 1.67× | 3.07× |
| saliency | 2.04× | 2.02× | 2.39× |
| *energy detector (control)* | *1.86×* | *2.06×* | *2.25×* |
| *random (control)* | *1.03×* | *1.03×* | *1.03×* |

The random control sits at chance, and enrichment increases monotonically as the burst gets
louder — the degradation control a localisation metric needs in order to be believed. The
attribution methods beat the energy detector by roughly 2.8× at the strongest burst;
saliency barely beats it and input × gradient starts *below* chance.

For comparison on the same stimulus and metric, the **original pipeline scores at chance**:
1.04× enrichment, with a map whose coefficient of variation is 0.064 — a constant plane.
That is the quantitative form of "the results were not super promising": the maps carried no
information, so there was nothing to interpret.

## The finding that came out of the fixed pipeline

With attribution working, a much sharper question becomes answerable: how does the mask
decision at output frequency `f_out` depend on input frequency `f_in`? The coupling matrix
`R[f_in, f_out] = sum_t |dz(t0, f_out) / dL(t, f_in)|` is exactly computable — 257 backward
passes on one forward graph, no baselines, no completeness axiom, no Captum hooks, and no
need to decompose the GRUs. It is therefore immune to every defect above.

On real speech at 5 dB SNR, in the regime where the model's VAD is demonstrably engaged
(mean mask gain 0.199 on speech frames against 0.016 on pauses, a 12.4× ratio):

| system | top-1 singular value | participation ratio | diagonal enrichment |
| --- | --- | --- | --- |
| NsNet2 mask logit (3 frames) | 0.946–0.992 | 1.02–1.12 | 1.04–1.42 |
| oracle Wiener gain (control) | 0.014–0.019 | 111–126 | **15.37** |

`R` is essentially **rank one**. That means `R ≈ a[f_in] ⊗ b[f_out]`: the shape of the
input-evidence profile is the same for every output frequency, and per-frequency differences
amount to a single scalar. NsNet2's mask is a **global speech-presence gate, not a per-bin
SNR estimator**. The oracle Wiener control — diagonal by construction — is measured through
the identical pipeline and comes out strongly local, so the apparatus provably detects
per-bin behaviour when it is there.

This also explains a puzzle in the table above: a narrowband burst 20 dB above the speech
barely changes the mask *inside its own band* (0.788 against 0.820 overall). The model does
not notch out interferers; it decides whether speech is present and gates broadband.

Two supporting measurements: 39.3% of pre-sigmoid mask logits have |z| > 4.6, where the
sigmoid derivative is below 0.01, and the median |z| is 4.07 — more than a third of the
time-frequency plane is gradient-dead on the mask *output*, which is the empirical
justification for targeting logits. And all attribution methods agree with a brute-force
occlusion sweep at Spearman 0.81–0.90.


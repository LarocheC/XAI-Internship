# Uncertainty-gated suppression: don't damage the signal where the model doesn't know

The idea (Clement's): rather than adapting the enhancer in the field — which the
`dnsmos_exported` study showed gets its quality judge fooled within one step — track
*uncertainty* and back the suppression off where the model is likely to be doing harm.

It needs no adaptation, no judge, no gradients, no weight updates, and it fails safe toward
the unprocessed input. Forward-only, so it runs inside the existing inference budget.

## Mechanism

Per time-frequency bin, predict the **over-suppression risk** `u[f,t]` — the probability that
this bin is speech-dominated (`|clean| > |noise|`) *and* being suppressed — then release the
gain toward identity in proportion:

```
g'[f,t] = g[f,t] + u[f,t] · (1 − g[f,t])
```

`u` is read forward-only from the model's own internal state: a 64-d slice of the GRU2 hidden
state plus the bin's own gain, log-power and normalised frequency — **67 features**, a
logistic regression. Trained offline on anechoic audio; no reference is needed at run time.

## Result

24 held-out LibriSpeech speakers × 4 noise types × 2 SNRs = 192 utterances, **reverberant**
(RT60 0.45 s), with the probe **trained on anechoic audio only** — so it must generalise to a
shift it never saw, which is the deployment condition.

| system | PESQ | ESTOI | SI-SDR | ΔPESQ |
| --- | ---: | ---: | ---: | ---: |
| noisy input | 1.411 | 0.685 | 2.50 | −0.327 |
| model, unmodified | 1.738 | 0.707 | 5.62 | — |
| global `g^0.7`, floored | 1.837 | 0.716 | 5.79 | +0.099 |
| global `g^0.5`, floored — *the constant knob* | 1.878 | **0.729** | 5.83 | +0.140 |
| **uncertainty-gated (67-d probe)** | **1.926** | 0.710 | 5.70 | **+0.188** |
| *gated by random u (control)* | 1.800 | 0.703 | 5.65 | +0.062 |
| *gated by oracle risk (ceiling)* | **2.378** | **0.828** | **6.79** | **+0.640** |

**The oracle establishes the headroom: +0.640 PESQ, +0.121 ESTOI, +1.17 dB SI-SDR.** Knowing
per-bin over-suppression risk is worth a great deal under shift. That is the result that makes
the direction worth pursuing, independently of how good any particular probe is.

**A 67-feature probe beats the constant knob on PESQ** (+0.188 vs +0.140) and beats the
matched-strength random control 3× (+0.062). It captures **29%** of the oracle headroom.

## The matched-release control — and the resolution of the ESTOI puzzle

Softening a mask always trades noise removal against speech preservation, and the two methods
release different amounts of gain, so a single-setting comparison cannot show the probe is
*smarter* rather than merely at a luckier operating point. Sweeping both over release strength
and comparing at matched release (mean `g' − g`), `diagnostics/uncertainty_matched_release.py`,
64 clips per point:

| mean release | global knob | **probe** | random | oracle |
| ---: | --- | --- | --- | --- |
| 0.020 | P=2.014 E=0.692 | **P=2.191 E=0.695** | P=2.079 E=0.695 | P=2.602 E=0.785 |
| 0.037 | P≈2.03 E≈0.691 | **P=2.241 E=0.691** | P=2.083 E=0.691 | P≈2.83 E≈0.817 |

**The probe curve dominates the global curve throughout the measurable range, on both metrics.**
It reaches PESQ 2.241 while releasing 0.037 of gain; the global knob needs 0.149 — **4× more
release** — to reach only 2.200, and degrades beyond that (2.043 at 0.282). *Where* the gain is
released matters more than how much.

**This resolves the ESTOI disagreement.** It was an operating-point artefact: the global knob's
ESTOI advantage (0.729 vs 0.710 in the single-setting table above) came entirely from operating
at ~4× the release the probe could reach. At matched release 0.020 the probe is *better* on
ESTOI as well (0.695 vs 0.692). The earlier caveat is withdrawn.

**Limit of the comparison.** The probe's release saturates at 0.037, because `u ∈ [0,1]` and the
strength multiplier tops out at 1.0. Every matched-release row above 0.037 is flat extrapolation
by `np.interp`, not measurement, and must not be quoted. The valid region is release ≤ 0.037.
Letting the probe reach higher release (rescaling `u`, or a different correction form) is
untested and is the obvious next design step.

## Remaining caveats
* **On matched (anechoic) conditions there is no headroom at all**: an earlier run of this
  experiment found the unmodified model best and every softening variant worse. The gain exists
  only under shift, which is consistent with the eco8/dnsmos reverb study finding a constant
  knob worth +0.64 there and adaptation pointless on matched data.
* **Synthetic reverb** (exponential-decay random IR), not measured RIRs. n=192, one model
  (NSNet2), no confidence intervals yet.
* The first version of this experiment was **mis-designed** — risk was collapsed to a per-frame
  gate, so a flagged frame released its whole spectrum, and the "oracle" scored *below* the
  probe. A correct-answer signal underperforming an approximation is the diagnostic that the
  rule, not the signal, is broken. Per-bin gating fixed it and turned the oracle into a real
  ceiling. Kept here because the failure is instructive.

## The capacity-fair comparison — and a correction

An earlier claim in this branch, that internal state beats output-level signals by 2.8× on AP,
used a **4-dimensional pooled** mask summary as the output baseline. That understates the
output: it is a pooled view of a 257-bin surface. Given the output at full resolution
(`diagnostics/fair_output_baseline.py`, NSNet2, 815k train / 1.04M test bins, held out by
speaker):

| features | dim | AUC | AP | uses |
| --- | ---: | ---: | ---: | --- |
| own gain only | 3 | 0.796 | 0.390 | output |
| mask across ±8 bins | 19 | 0.806 | 0.386 | **output only, full resolution** |
| **64-d GRU state + own gain** | 67 | **0.844** | **0.473** | internal |
| both | 83 | 0.848 | 0.469 | — |
| chance | — | 0.500 | 0.077 | — |

**Internal state beats a fair output baseline by ΔAP +0.087** — not 2.8×. The 2.8× figure is
withdrawn.

The corrected effect replicates independently: an adversarial review of this work measured
ΔAP **+0.078, 95% CI [+0.043, +0.111]** on *LiSenNet* (24-d bottleneck tap vs the full 514-d
output), with a paired bootstrap over clip blocks, and found the margin *grows* to +0.100 under
a nonlinear readout. Different model, different experimenter, same answer to within 0.01 AP.

Two further readings from the table. Extra output resolution does not rescue the output
baseline on this model (19-d scores 0.386 against 3-d's 0.390) — it saturates. And adding the
output on top of internal state adds nothing (0.469 vs 0.473), so internal state subsumes it.

### The scope condition, which the paper must state

Adverse prior art is explicit that internals usually add little over the output *when the
output is available in full* — in one verifiable-reward-hacking study a linear probe reached
AUC 0.998 while string-matching the output reached 100%. **Internals win where the output has
been discarded.** On a streaming device that is exactly the situation: the full 257-bin mask
history is not retained, while the recurrent state is, by construction, already there.

### A structural limit: this cannot work on ConvFSENet

ConvFSENet's head is `Conv1d(192 → 257, k=1)` + sigmoid. Measured from the published
checkpoint: numerical rank **192 of 192**, σ_max 4.43, σ_min 0.0209, **condition 212** — full
column rank and well conditioned. So `z = Wh + b` determines `h` exactly (`h = W⁺(z − b)`) and
the sigmoid is elementwise invertible: **the deployed mask determines the 192-d residual state
exactly, and the internal state carries no information the output lacks.** The hypothesis is
structurally false for that model. It survives on LiSenNet and NSNet2, whose masks are produced
through non-invertible paths — check the head's conditioning before assuming it holds anywhere
else.

## Why this is the strongest direction in the branch

It answers the original question — *could some method actually run on my MCU and produce
results that are useful?* — with yes, and with a number. It is forward-only and tiny (the probe
is ~400 MAC/frame at the frame level, and the per-bin version reuses tensors already exported
as FIFO graph outputs). It beats the constant baseline that beat everything else. It needs
none of the machinery that the on-device adaptation study found fragile. And the failure it
prevents — speech damage under unseen conditions — is the one a hearing-device company cares
about most.

## Next, in order

1. **Let the probe reach higher release.** Its curve is still rising where it saturates at
   0.037, so the measured advantage may understate it. Rescale `u` or change the correction form.
2. **Close more of the gap to the oracle.** The probe is deliberately minimal; a small MLP, more of the
   state, and temporal context are all untried.
3. **Confidence intervals and more shifts** — measured RIRs, the four conflicting distortions
   of the mixed-shift testbed, and DNSMOS SIG/BAK separately, since SIG is the metric this is
   meant to protect.
4. **Replicate on LiSenNet and ConvFSENet**, where 92% of activations are already exported.
5. **Cost it on the STM32N6** against the measured 13.21 ms of free budget.

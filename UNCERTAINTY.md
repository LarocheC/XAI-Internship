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

## Caveats, before anyone builds on this

* **ESTOI disagrees with PESQ.** The global knob scores 0.729 against the probe's 0.710 while
  losing on PESQ. The advantage is metric-specific and unexplained. A plausible story — PESQ
  rewards the probe's sparse targeted release, ESTOI prefers broad softening — is a hypothesis,
  not a finding. Resolve this before claiming the probe wins.
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

## Why this is the strongest direction in the branch

It answers the original question — *could some method actually run on my MCU and produce
results that are useful?* — with yes, and with a number. It is forward-only and tiny (the probe
is ~400 MAC/frame at the frame level, and the per-bin version reuses tensors already exported
as FIFO graph outputs). It beats the constant baseline that beat everything else. It needs
none of the machinery that the on-device adaptation study found fragile. And the failure it
prevents — speech damage under unseen conditions — is the one a hearing-device company cares
about most.

## Next, in order

1. **Explain the ESTOI/PESQ split.** Until that is understood the headline is not safe.
2. **Close more of the 71% gap.** The probe is deliberately minimal; a small MLP, more of the
   state, and temporal context are all untried.
3. **Confidence intervals and more shifts** — measured RIRs, the four conflicting distortions
   of the mixed-shift testbed, and DNSMOS SIG/BAK separately, since SIG is the metric this is
   meant to protect.
4. **Replicate on LiSenNet and ConvFSENet**, where 92% of activations are already exported.
5. **Cost it on the STM32N6** against the measured 13.21 ms of free budget.

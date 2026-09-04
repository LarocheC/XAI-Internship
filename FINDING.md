# The gain surface is low-dimensional — and lower than its own training target

This replaces the withdrawn rank-one coupling result (`RETRACTION.md`). It uses a
**forward-only** statistic: no Jacobian, no backward pass, nothing that can be contaminated by
taking an absolute value before a sum.

## Statistic

`G[f,t] = σ(z[f,t])`, the model's own predicted gain surface. Effective rank is reported as
`r90` / `r99` — the number of singular values of the mean-centred `G` holding 90% / 99% of
spectral energy. Top-1 singular value is **not** reported: a uniformly random surface scores
0.75 on it, so it carries almost no information here.

**The estimator passes a known-rank recovery test** (`diagnostics/gain_rank_null.py`), which is
the check the withdrawn statistic failed. Surfaces built with rank `k` read back as:

| constructed k | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 257 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r90` read back | 1 | 2 | 4 | 7 | 13 | 24 | 42 | 65 | 85 |

Monotone across the whole range, and unchanged when the surface is passed through a sigmoid.

## Measurement

Six LibriSpeech test-clean utterances, white noise, `diagnostics/gain_rank.py`:

| system | r90 @ 0 dB | r90 @ 5 dB | r90 @ 15 dB | **r99 @ 5 dB** |
| --- | ---: | ---: | ---: | ---: |
| **NSNet2 predicted gain** | 4.7 | 4.7 | 3.3 | **18.3** |
| oracle IRM — *the training target* | 22.8 | 22.2 | 21.2 | **86.7** |
| oracle Wiener gain | 28.5 | 28.2 | 27.8 | 83.8 |
| shuffled NSNet2 *(null)* | 107.8 | 108.5 | 109.3 | 160.7 |
| random gain *(null)* | 109.3 | 109.7 | 109.3 | 161.5 |

Two independent nulls agree at ~109, so the statistic reads time-frequency structure rather
than the marginal distribution of gain values. The ceiling — the oracle masks the model was
trained to approximate — sits at 22–28 (`r90`) and 84–87 (`r99`). The model sits far below both.

## Causal test

Truncate `G` to rank `r`, re-apply, resynthesise (`diagnostics/gain_rank_truncation.py`):

| gain surface | SI-SDR (dB) | LSD (dB) |
| --- | ---: | ---: |
| noisy input | 5.01 | 33.58 |
| full model | **13.99** | 13.82 |
| rank 4 | 12.16 | 23.67 |
| rank 8 | 13.30 | 17.52 |
| rank 16 | 13.81 | 15.29 |
| rank 32 | 13.96 | 14.55 |
| *random rank-4, matched energy* | *−0.36* | 22.55 |
| *random rank-16, matched energy* | *−0.33* | 22.57 |

**The random-subspace control is the important row.** A random rank-16 surface scores −0.35 dB,
*worse than not processing at all*. The subspace the model uses is specific and learned; the
result is not "any smooth gain works".

**And it calibrates the statistic.** Rank-4 truncation costs 1.83 dB, so `r90 = 4.7` overstates
the compressibility. Quality saturates at rank 16–32, which matches `r99 = 18.3`. **`r99` is the
statistic to quote**; `r90` is not perceptually calibrated.

## The claim

*A trained mask-based speech enhancer produces a gain surface roughly five times
lower-dimensional than the ideal ratio mask it was trained to approximate (r99 ≈ 18 vs ≈ 87);
truncating to the model's own effective rank costs ≈ 0.2 dB, while a random subspace of the same
rank destroys the output.*

## What is still owed

* **Cross-architecture replication** on LiSenNet and ConvFSENet. This is a requirement, not a
  bonus: the likely reviewers (Amar/Ivry/Cohen) replicate across three models as a matter of
  course, and a claim shown only on a 2020 GRU baseline reads as a property of one model.
* **Corpus scale with real noise.** Six utterances of white noise is a pilot. VoiceBank-DEMAND
  gives 824 utterances over five unseen noise types at four SNRs, already in the eco8 harness.
* **Confidence intervals** on every number above, bootstrapped over utterances.
* **PESQ/DNSMOS** alongside SI-SDR and LSD — the eco8 metric harness already computes them.
* **The int8 question.** The statistic is forward-only, so it is computable on the deployed
  quantised graph. Whether `r99` moves under quantisation is an open and separately interesting
  question, and the prior-art sweep found nobody computing attribution through an integer
  inference graph.

## Provenance note

Two estimators were withdrawn before this one: a diagonal-band metric whose width scaled
silently with sampling density, and the coupling statistic of `RETRACTION.md`. Both were caught
by controls rather than by reading the code. Every statistic here ships with the null that
licenses it.

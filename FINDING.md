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

## The claim — NARROWED after cross-architecture replication

The class-level claim first written here ("a trained mask-based enhancer produces a gain surface
roughly five times lower-dimensional than the ideal ratio mask it was trained to approximate")
is **false**. Replicating on LiSenNet with the oracle computed in its own gain domain:

| system | params | r90 | r99 | own IRM r90 | own IRM r99 | random null r90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NSNet2 | 2,783,657 | **4.7** | 18.3 | 22.2 | 86.7 | 109.7 |
| LiSenNet `conv-hardened` | 36,288 | **41.3** | 127.2 | 22.2 | 112.5 | 109.2 |

LiSenNet's gain surface is **higher**-dimensional than its own oracle target — the reverse of
NSNet2, and nearly an order of magnitude above it. Whatever NSNet2 is doing, it is not a
property of mask-based enhancement.

What survives is the *variation*, which is a weaker but real observation:

*The effective rank of a trained enhancer's gain surface varies by nearly an order of magnitude
across architectures (r90 4.7 to 41.3) and is not predicted by the oracle target, which is
nearly identical for both (r90 22.2). For NSNet2 the surface is far below its target and
rank-16 truncation is perceptually free; a random subspace at matched rank destroys the output.*

**The interesting direction this opens.** The two models sit at opposite ends of both rank
*and* redundancy: NSNet2 has 77× more parameters, a rank-4.7 output, and compresses 24× at no
PESQ cost (2.837 vs 2.845 for `monarch_40` at 117k); LiSenNet is already tiny, has a rank-41
output, and has no comparable headroom. n=2 is an anecdote, not evidence — but the hypothesis
*gain-surface rank tracks parameter redundancy* is now testable with a **validated** statistic
across the 27 public `sparse-nsnet2-checkpoints` variants within a single family, where the
architecture is held fixed and only the factorisation changes.

ConvFSENet could not be instantiated standalone (`ValueError: unsupported norm_type for the
standalone: cLN`) and is still owed as the third architecture.

### That hypothesis has now been tested too, and it fails

Within-family sweep over the public eco8 NSNet2 checkpoints — architecture, training recipe,
dataset, STFT grid and oracle all held fixed by construction; only width changes
(`diagnostics/gain_rank_family_sweep.py`):

| variant | params | r90 | r99 |
| --- | ---: | ---: | ---: |
| `dense_h52` | 77,087 | 2.0 | 4.3 |
| `dense_h68` | 117,863 | 2.0 | 3.8 |
| `dense_h100` | 223,607 | 2.0 | 4.3 |
| `dense_h148` | 442,703 | 2.0 | 3.2 |
| `dense_h216` | 877,325 | 2.0 | 5.7 |
| `baseline` | 2,783,657 | 2.0 | 4.7 |

`r90` is **constant at 2.0 across a 36× range of parameter count**. Spearman(params, r90) is
undefined because the statistic does not vary; `r99` gives +0.493, the opposite sign to the
prediction and not significant at n=6. **Gain-surface rank does not track parameter
redundancy.** The hypothesis is withdrawn.

Two limits on how much this result can bear:

* **The factorisation arm is missing.** The six `blockdiag_*` and `monarch_*` checkpoints — the
  axis that maps most directly onto redundancy — failed to load (`ImportError:
  torch_structured.monarch.BlockdiagLinear`; the installable `torch_structured` does not expose
  that name). Only the width ladder was tested, and width is a weaker proxy for redundancy
  than factorisation.
* **`r90 = 2.0` exactly, six times over, is itself a warning.** A statistic pinned at the same
  integer for every member of a family is more likely at a floor than measuring something. It
  should not be read as "these models all have rank-2 gain surfaces" without first establishing
  that `r90` can move at all on this family — e.g. by checking that it responds to SNR, noise
  type and utterance within a single checkpoint.

This is the third successive hypothesis in this branch to die at its first proper control.

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

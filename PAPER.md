# Paper direction: the effective rank of a learned time-frequency mask

Target: STM32N6. Goal: a novel, publishable result. This note records what is measured, what
the prior-art sweep found, and which of three candidate theses to back.

## The finding

The input-frequency × output-frequency coupling of the mask decision,
`R[f_in, f_out] = Σ_t |∂z(t₀, f_out) / ∂u(t, f_in)|` where `z` is the **pre-sigmoid** mask and
`u` the network's magnitude-like input, has a singular spectrum that differs sharply by
architecture — and an oracle per-bin Wiener gain, measured through the identical code, is
strongly local, so the instrument demonstrably detects per-bin behaviour when it exists.

| system | params | PESQ (FP32) | top-1 SV | participation ratio | diag enrichment |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle Wiener gain *(positive control)* | — | — | 0.081 | 23.08 | **15.42** |
| NSNet2 | 2,783,657 | 2.845 | **0.925** | **1.17** | 1.59 |
| LiSenNet `conv-hardened` (3-channel input) | 36,288 | 3.013 | 0.537 | 3.25 | 2.98 |
| LiSenNet `conv-hardened` (magnitude channel only) | 36,288 | 3.013 | 0.466 | 4.20 | 3.10 |

All rows: `n_out = 64` sampled output frequencies, `F_in = 257`, same code, same stimulus, same
frame, LibriSpeech utterance at 5 dB SNR. Reproduce with
`diagnostics/coupling_matrix.py` and the cross-model script in the scratchpad.

**NSNet2's mask decision is essentially rank one** — a global speech-presence gate.
**LiSenNet's is not**, despite being 77× smaller and scoring better (3.013 vs 2.845 PESQ). The
75×-smaller model uses a measurably richer and more frequency-local rule than the large one.
Neither resembles the per-bin Wiener control.

### Two traps already sprung, and what they cost

The first version of the diagonal metric squared `R` by subsampling its rows, which silently
rescaled the ±8-bin band with `n_out`: the *same* oracle Wiener system read enrichment 15.37 at
`n_out=257` and 3.10 at `n_out=48`. Every cross-model comparison made with it was worthless.
Fixed by scoring the rectangular matrix directly with the band in original bin units; the
control now reads 15.42 / 15.35 / 15.37 at `n_out` = 64 / 128 / 257. **Top-1 SV and
participation ratio remain `n_out`-dependent by construction** (PR is bounded by
`min(F_in, n_out)`), so those two are quotable only at matched settings.

Second, LiSenNet takes three input channels (compressed magnitude, group delay, IFD) where
NSNet2 takes magnitude alone, so differentiating through all three would confound architecture
with input representation. The magnitude-only row exists to close that: it moves the numbers
very little, so the difference is architectural.

The general lesson, and it belongs in the paper: **a structural claim about a model needs a
synthetic system with a known answer measured through the identical pipeline**, or a finding
cannot be distinguished from an artefact of the instrument.

## What the prior-art sweep found

**The rank claim is open.** Hard negatives, not soft ones: arXiv metadata returns zero for
`speech enhancement` with `effective rank`, with `intrinsic dimension`, and for
`participation ratio` with `speech`. Caveat recorded honestly — arXiv searches metadata, not
full text.

Sivasankaran, Vincent & Fohr, *Explaining Deep Learning Models for Speech Enhancement*
(Interspeech 2021) is the **parent, not the scoop**, and the distinction survives review: they
construct exactly this input-bin × output-bin coupling tensor, then collapse it to a scalar
speech-relevance mass ratio that is **provably blind to rank** — a rank-one global gate and a
perfectly diagonal per-bin estimator can score identically. Nobody has looked at the spectrum.
That paper has 9 citations in five years, only 2 of them on enhancement (both Technion,
Amar/Ivry/Cohen). The Interspeech 2025 interpretability tutorial contains no enhancement, no
regression and no masking content at all. Those two facts establish the gap without hand-waving.

**Closed, do not build here.** "XAI relevance drives pruning / mixed-precision quantisation" is
saturated: ECQx (Becking/Samek/Lapuschkin 2022), GMPQ (ICCV 2021 / IJCV 2024), HAWQ-V2,
Cuttlefish (MLSys 2023), Swift-SVD (2026). Landing in those cells reads as engineering.

**Open, and adjacent to us.** (i) *Attributing the FP32-vs-int8 output delta* — every existing
paper compares explanations across precisions or regularises toward FP32 attributions; nobody
treats the precision-induced delta as a signal to be explained. (ii) *Rank of the input-output
coupling as a predictor of structural factorisability* — existing predictors are all weight-space
spectral statistics (data-independent) or per-layer activation statistics, never a model-level
Jacobian quantity, and none is validated across *different structured formats* at matched budgets.

**On-device XAI is real but weak as a flagship.** No published work runs an explanation method on
a commercial edge NPU with silicon-measured latency and memory — the nearest, Bhat et al.
(VLSI-SoC 2022), is simulated FPGA HLS at 100 MHz, 16-bit, vision, and the explanation is never
used for anything. But with 82% of the 16 ms hop free on LiSenNet, "it fits" carries no tension.
The calibration signal is blunt: an ICASSP 2026 submission that inspects an enhancer's internals
and shows pictures was placed in the **demo track**, not the technical programme.

**Two things to verify personally.** The sweep flags an ICASSP 2026 accepted paper, *From Diet to
Free Lunch: Estimating Auxiliary Signal Properties using Dynamic Pruning Masks in Speech
Enhancement Networks*, as being from your own group and as already occupying the "what does an
enhancer's internal state secretly know" slot — which partly self-scoops the runtime-probe thesis.
Confirm that. And Amar/Ivry/Cohen are the most likely reviewers; they deliberately replicate
across MUSE, MP-SENet and Demucs, so **cross-architecture replication is a requirement, not a
bonus**. A rank claim shown only on NSNet2 — a 2020 GRU baseline — will be read as a property of
one model.

## The thesis to back

**"The mask is a gate: effective rank of learned time-frequency suppression."** The spine is the
table above, extended to a corpus with confidence intervals and at least three architectures
(NSNet2, LiSenNet, ConvFSENet), always against the oracle-Wiener positive control.

What elevates it above a measurement paper is the falsifiable prediction: **low effective rank
means high redundancy, so rank should predict how far a model can be structurally factorised
without quality loss.** NSNet2 is rank ≈ 1.17 and your own sweep already factorises it 2.78M →
117k with PESQ unchanged (2.837 vs 2.845, `monarch_40`); LiSenNet is rank ≈ 3–4 at 36k params and
should resist compression. That is testable in-house **without training anything** —
`claroche1/sparse-nsnet2-checkpoints` holds 27 public variants: a `dense_h52 … dense_h216`
capacity ladder plus full `blockdiag`, `monarch` and `butterfly` families, all with measured PESQ.

The STM32N6 then supplies the payoff rather than the headline: the compression the diagnostic
predicts is the compression that fits on-chip, and the deployed-vs-trained rank on the int8 graph
is a genuine second contribution (no prior art computes attribution through an integer graph).

### The order to run it in

1. **Negative control first.** Measure an untrained NSNet2. If random weights also read rank one,
   the finding is about the architecture's forward map, not about what it learned — stop and
   re-scope. This is one afternoon and it is the cheapest way to kill the paper.
2. **Corpus scale.** Replace the single utterance and single frame with ~300 utterances, frames
   stratified speech/pause, bootstrap CIs, across SNR and noise type.
3. **Third architecture.** ConvFSENet (TCN), tapped at its pre-sigmoid backend output.
4. **The compressibility correlation** across the 27 checkpoints — the experiment that turns a
   measurement into a prediction.
5. **Non-gradient confirmation.** The Jacobian is a gradient object and ~39% of the mask plane is
   saturated; one FreqRISE-style masking measurement that owes nothing to gradients.
6. **Deployed rank** on the int8 graph, and the board numbers.

### Timing

**ICASSP 2027 closes 16 September 2026 — twelve days out**, and DATE 2027 on 20 September. Neither
is realistic: steps 1–4 are the paper, and step 1 alone could kill it. Interspeech 2027
(São Paulo, deadline unannounced, historically late February / early March 2027) is the right
target, with EUSIPCO 2027 as the fallback. That gives roughly six months — enough to do it
properly, which matters when the likely reviewers replicate across three models as a matter of
course.

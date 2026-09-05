# On-device explainability: the measured budget, and what it rules out

Grounded in `larochec/eco8-neaixt`'s own measurements, re-derived from source where the repo's
figures disagree with each other. Provenance labels follow that repo's convention: **SILICON**
(board, capture committed), **ISS** (Cadence simulator), **MODELLED** (datasheet arithmetic).

## Corrections to figures I quoted earlier in this session

Four of the numbers I stated were wrong or imprecise. Recorded here rather than quietly fixed:

* **"NSNet2 `blockdiag_full`: 2.13 ms, RTF 0.13 on the STM32N6" — wrong.** That row is
  `monarch_full` (`docs/targets/stm32n6.md:37`). **No `blockdiag_full` figure exists for the
  STM32N6 at all.** Every file under `deploy/stm32n6/` also still uses pre-rename labels: what
  `ONBOARD_MEASUREMENT.md:103-104` calls `monarch_full`/`monarch_8` are `blockdiag_full`/
  `blockdiag_8`, and genuine two-factor Monarch has never run on that chip.
* **"1.24 MB DSP data segment" is one of three mutually inconsistent figures** in the repo
  (1.24 MB, 1.30 MB, and a third), none derived from a committed linker script. The usable
  window is ~1,280 KiB; treat the exact ceiling as unestablished.
* **ConvFSENet on HiFi4 is 2,812,605 cyc/frame**, not the 2,767,999 in the older table — the
  original was measured with a dead recurrent state (`results/PROVENANCE.md:57-58`).
* **LiSenNet nc24's "47 KB weights"** is the ST tool's blob size including its own constants
  and alignment. The model is 36,288 parameters = **35.6 KiB** of int8 weights.
* I implied a tension between "17 FIFO states" and "157,830 state elements". There is none:
  157,830 / 25 states is `relu6-deep`; the deployed nc24 is **17 states / 57,990 elements**.

Also worth knowing before quoting anything: **the repo contradicts itself on STM32N6
provenance.** `docs/targets/stm32n6.md:39-46` labels all six N6 rows SILICON;
`docs/README.md:88-92` refuses that label for the same rows because no raw capture is committed
anywhere under `deploy/stm32n6/`. Treat them as board-measured-but-uncaptured.

## The actual budget

| target | passing row | used | **free** |
| --- | --- | --- | --- |
| RT595 HiFi4 | NSNet2 `blockdiag_full` | 1,466,196 cyc / 7.405 ms / 46.3% duty | **1,701,804 cyc, 8.595 ms, ~528 KiB data SRAM** (SILICON) |
| STM32N6 NPU | LiSenNet `conv-hardened` nc24, 17-state | 2.79 ms, 194 KiB | **13.21 ms, ~2.6 MB on-chip** |

In useful units: the RT595 leftover buys **1.16 extra forward passes per frame**; the STM32N6
buys **4.73**. There is also a **second, entirely unbudgeted resource on the RT595** — the M33
sits idle during DSP inference and can run a kernel in parallel at zero DSP cost, though 3.63×
slower per operation.

**Memory binds more often than compute.** Dense NSNet2 is at 0.48× the cycle budget but 2.27×
over the data segment; ConvFSENet is inside its cycles but ~1.5× over; `monarch_full` dies on a
1,169 KiB *arena* while within 1.04× of the cycle budget. On the N6, dense NSNet2 is explicitly
memory-bandwidth-bound (2.81 MB/frame from octoFlash at 122 MB/s ≈ the whole 22.94 ms). **The
design rule: a kernel that materialises a large intermediate is more dangerous than one that
burns cycles.** A 257×257 float coupling matrix is 264 KiB — half the RT595's spare SRAM.

## What this rules out

**Gradient-based attribution is dead on-device, by one to two orders of magnitude.** A backward
pass over LiSenNet nc24 on the RT595 HiFi4 is ≈3 × 1,002,566 MACs at its measured 6.25 cyc/MAC
= **18.8 M cycles/frame — 5.9× over the entire budget**, before the forward pass. The memory
argument kills it independently. DeepLIFT, DeepLiftShap, integrated gradients and saliency are
all out. This is the quantitative vindication of the forward-only direction in `FINDING.md`,
and it is a stronger argument than "the NPU is inference-only".

**A parameter-count intuition I was using is false.** "LiSenNet is 19× smaller than NSNet2
`blockdiag_full`, so an explanation on it is 19× cheaper" is wrong. Counted independently by
forward hooks (`diagnostics/count_macs.py`):

| model | parameters | MACs/frame | MAC/param |
| --- | ---: | ---: | ---: |
| LiSenNet `conv-hardened` nc24 | 36,288 | **900,894** (hooks) | **24.8×** |
| NSNet2 `blockdiag_full` | 701,657 | 701,657 (analytic) | 1.0× |

**19.3× fewer parameters, 1.28× more per-frame arithmetic.** The mechanism is the point: a
convolutional model reuses each weight across frequency positions, so MACs ≫ parameters, while
an FC+GRU model uses each weight once per frame, so MACs ≈ parameters. Probes and backward
passes are priced in MACs, so parameter count is the wrong unit for costing an explanation.

(The grounding agent reported 1,002,566 MACs/frame and a 1.44× ratio. My hook count gives
900,894 and 1.28× — an 11% disagreement, probably a difference in what is counted for the GRU
gates and the sub-pixel upsampling. The direction and the mechanism are unaffected; use the
hook figure, or better, re-count against the deployed graph.)

**Full state telemetry is impossible.** The deployed nc24's state stream is 57,990 int8
elements/frame = **3.62 MB/s**. Nothing like that leaves a hearing device.

## What it permits

**92% of LiSenNet's per-frame activations are already free.** The 17-state streaming export
emits all FIFO buffers as graph outputs, and a FIFO's newest column *is* the current frame's
activation — 57,990 of 63,260 elements across **seven distinct tap points** (encoder stages 1–3,
the DPR inter-time chain at four depths, the GLU gate input, and the final pre-mask 6×256 map),
with per-tensor scales already baked into the generated header. Reading them costs nothing.

Not exposed, and needing a re-export: the DPR stack output, the intra-frequency path, and the
pre-sigmoid logit. ConvFSENet's free taps are 4.7× smaller but are the *wrong* ones — the
192-wide residual stream that carries the signal is invisible.

That is the concrete basis for the probe result in `REVIEW.md` (400 MACs, AP 0.468 vs 0.298 for
the best no-internals baseline): on this build the tap is free, the scales are already there,
and the arithmetic is negligible against 13.21 ms of headroom.

## Caveat

These figures come from agents reading the eco8 repo and instantiating its models; the MAC
counts and state volumes were re-derived from source, but I have not independently verified
every one against hardware. The corrections above are exactly the kind of error they are
correcting, so treat this file as a starting point for your own check rather than as settled.

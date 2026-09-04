"""Reproduce the attribution experiment with a known right answer.

A narrowband noise burst is injected into clean speech at a chosen frequency and time.
NsNet2 suppresses it, and we ask each attribution method the same question -- "which parts
of the input drove the suppression decision inside the burst region?" -- for which the
ground truth is known by construction. Methods are scored on how much attribution mass
lands in the injected region (``enrichment``, where 1.0 is chance), on the pointing game,
and on agreement with a brute-force occlusion sweep.

Usage::

    python DeepShap/validate_attributions.py --input path/to/clean.wav
    python DeepShap/validate_attributions.py            # synthesises a stimulus instead
"""

import argparse
import json
import os
import sys
import wave

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr

from config.parameters import n_fft, sample_rate
from utils.mask_attributions import (
    attribute_region,
    control_maps,
    localisation_scores,
    noise_floor_baselines,
    occlusion_reference,
)
from utils.model_utils import load_nsnet2_model

METHODS = ["deeplift_shap", "deeplift", "integrated_gradients", "gradient_shap",
           "input_x_gradient", "saliency"]


def read_wav(path, target_sr=sample_rate, max_seconds=3.0):
    """Minimal 16-bit PCM reader.

    torchaudio.load requires the optional torchcodec backend from 2.9 onwards, and
    requirements.txt pins no versions, so a plain reader keeps this script runnable.
    """
    with wave.open(path, "rb") as handle:
        sr, n_channels = handle.getframerate(), handle.getnchannels()
        if handle.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM wav files are supported")
        raw = handle.readframes(handle.getnframes())
    audio = torch.frombuffer(raw, dtype=torch.int16).float() / 32768.0
    if n_channels > 1:
        audio = audio.view(-1, n_channels).mean(dim=1)
    if sr != target_sr:
        raise ValueError(f"expected {target_sr} Hz, got {sr} Hz")
    return audio[: int(max_seconds * target_sr)].unsqueeze(0)


def synthetic_speech(seconds=3.0):
    """A voiced/unvoiced alternation, when no real recording is available."""
    n = int(seconds * sample_rate)
    t = torch.arange(n) / sample_rate
    audio = 0.01 * torch.randn(n)
    for k, f0 in enumerate((140, 280, 420, 560, 700)):
        voiced = ((t * 3) % 1.0) < 0.6
        audio += (0.10 / (k + 1)) * torch.sin(2 * np.pi * f0 * t) * voiced
    return audio.unsqueeze(0)


def inject_burst(clean, centre_hz=3000.0, bandwidth_hz=300.0, t0=1.0, t1=1.6, snr_db=0.0, seed=0):
    """Add a narrowband noise burst at a stated SNR (dB) relative to the clean signal
    over the burst interval, so burst strength is a sweepable axis rather than an opaque
    amplitude ratio."""
    generator = torch.Generator().manual_seed(seed)
    n = clean.shape[-1]
    noise = torch.randn(1, n, generator=generator)
    spectrum = torch.fft.rfft(noise)
    freqs = torch.fft.rfftfreq(n, 1 / sample_rate)
    spectrum[:, (freqs < centre_hz - bandwidth_hz / 2) | (freqs > centre_hz + bandwidth_hz / 2)] = 0
    band = torch.fft.irfft(spectrum, n=n)
    envelope = torch.zeros(1, n)
    lo, hi = int(t0 * sample_rate), int(t1 * sample_rate)
    envelope[:, lo:hi] = 1.0
    speech_rms = clean[:, lo:hi].std().clamp(min=1e-9)
    band = band / band.std().clamp(min=1e-9) * speech_rms * (10.0 ** (-snr_db / 20.0))
    return clean + band * envelope


def region_slices(model, n_frames, centre_hz, bandwidth_hz, t0, t1):
    """TF support of a burst injected by inject_burst, which band-limits to
    centre +- bandwidth/2. Using +- bandwidth here made the scored region 19 bins wide
    where the burst is 9, inflating every localisation number."""
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate).numpy()
    half = bandwidth_hz / 2.0
    f_idx = np.where((freqs >= centre_hz - half) & (freqs <= centre_hz + half))[0]
    hop = model.preproc.hop_length
    t_lo, t_hi = int(t0 * sample_rate / hop), min(n_frames - 1, int(t1 * sample_rate / hop))
    return slice(int(f_idx[0]), int(f_idx[-1]) + 1), slice(t_lo, t_hi + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=None, help="16 kHz 16-bit mono wav; synthesised if omitted")
    parser.add_argument("--centre_hz", type=float, default=3000.0)
    parser.add_argument("--bandwidth_hz", type=float, default=300.0)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=1.6)
    parser.add_argument("--snr_db", type=float, default=0.0, help="burst SNR against the speech in the burst interval")
    parser.add_argument("--probe", type=str, default="displaced", choices=["displaced", "colocated"],
                        help="'displaced' explains a region away from the burst, so the "
                             "ground truth is not the region being explained; 'colocated' "
                             "explains the burst region itself and is CIRCULAR -- it scores "
                             "how much mass a method returns to the cell it was asked about, "
                             "which rewards diagonal attribution regardless of the model")
    parser.add_argument("--probe_hz", type=float, default=800.0, help="displaced probe centre frequency")
    parser.add_argument("--outdir", type=str, default="DeepShap/attributions/validation")
    parser.add_argument("--skip_occlusion", action="store_true")
    parser.add_argument("--patch", type=int, nargs=2, default=[16, 8])
    args = parser.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.outdir, exist_ok=True)

    model, device = load_nsnet2_model()
    model = model.to(device).eval()

    clean = (read_wav(args.input) if args.input else synthetic_speech()).to(device)
    clean = clean / clean.abs().max() * 0.5
    noisy = inject_burst(clean, args.centre_hz, args.bandwidth_hz, args.t0, args.t1,
                         snr_db=args.snr_db)

    log_power = model.log_power(model.preproc(noisy))                 # [1, F, T]
    n_freq, n_frames = log_power.shape[1], log_power.shape[2]
    # Ground truth = where the burst actually is.
    gt_f, gt_t = region_slices(model, n_frames, args.centre_hz, args.bandwidth_hz, args.t0, args.t1)
    ground_truth = np.zeros((n_freq, n_frames), dtype=bool)
    ground_truth[gt_f, gt_t] = True

    # The region whose mask logit we explain. Keeping it separate from the ground truth is
    # the whole point: if they coincide, the metric rewards a method for returning mass to
    # the cell it was asked about, which is circular and penalises exactly the non-local
    # behaviour this model turns out to have.
    if args.probe == "colocated":
        f_slice, t_slice = gt_f, gt_t
    else:
        f_slice, t_slice = region_slices(model, n_frames, args.probe_hz, args.bandwidth_hz,
                                         args.t0, args.t1)

    print(f"spectrogram {n_freq} x {n_frames} frames")
    print(f"burst (ground truth): freq bins {gt_f.start}-{gt_f.stop - 1} "
          f"({args.centre_hz:.0f} Hz), frames {gt_t.start}-{gt_t.stop - 1}, SNR {args.snr_db:+.0f} dB")
    print(f"explained region    : freq bins {f_slice.start}-{f_slice.stop - 1}, "
          f"frames {t_slice.start}-{t_slice.stop - 1}   [probe={args.probe}]")
    if args.probe == "colocated":
        print("  WARNING: co-located probe -- ground truth IS the explained region, so these "
              "numbers measure self-attribution, not localisation.")
    print(f"ground-truth region covers {ground_truth.mean() * 100:.2f}% of the plane "
          f"(so enrichment 1.0 = chance)\n")

    with torch.no_grad():
        mask = torch.sigmoid(model.mask_logits(log_power))[0]
    inside, overall = float(mask[gt_f, gt_t].mean()), float(mask.mean())
    print(f"sanity: mean mask inside the burst region = {inside:.3f} vs {overall:.3f} overall "
          f"(local suppression = {overall - inside:+.3f})")
    if overall - inside < 0.05:
        print("  note: little LOCAL suppression. Consistent with the rank-one coupling result "
              "-- NsNet2 gates broadband on speech presence rather than notching out a "
              "narrowband interferer. Not a bug in the probe.")
    print()

    baselines = noise_floor_baselines(log_power)
    results, maps = {}, {}
    for method in METHODS:
        attribution, delta = attribute_region(model, log_power, f_slice, t_slice,
                                              method=method, baselines=baselines)
        scores = localisation_scores(attribution, ground_truth)
        scores["convergence_delta"] = delta
        results[method], maps[method] = scores, attribution
        print(f"  {method:22s} enrichment={scores['enrichment']:6.2f}x  "
              f"mass={scores['mass_fraction'] * 100:5.2f}%  pointing={str(scores['pointing']):5s}"
              + (f"  |delta|={delta:.3f}" if delta is not None else ""))

    for name, control in control_maps(log_power).items():
        scores = localisation_scores(control, ground_truth)
        scores["convergence_delta"] = None
        results[name], maps[name] = scores, control
        print(f"  {name:22s} enrichment={scores['enrichment']:6.2f}x  "
              f"mass={scores['mass_fraction'] * 100:5.2f}%  pointing={str(scores['pointing']):5s}"
              "   <- control")

    if not args.skip_occlusion:
        print("\n  running occlusion reference (this is the slow one)...")
        occlusion = occlusion_reference(model, log_power, f_slice, t_slice, patch=tuple(args.patch))
        scores = localisation_scores(occlusion, ground_truth)
        scores["convergence_delta"] = None
        results["occlusion"], maps["occlusion"] = scores, occlusion
        print(f"  {'occlusion':22s} enrichment={scores['enrichment']:6.2f}x  "
              f"mass={scores['mass_fraction'] * 100:5.2f}%  pointing={str(scores['pointing']):5s}")
        print("\n  rank correlation with the occlusion reference:")
        for method in METHODS:
            rho = spearmanr(np.abs(maps[method]).ravel(), np.abs(occlusion).ravel()).statistic
            results[method]["spearman_vs_occlusion"] = float(rho)
            print(f"    {method:22s} rho={rho:+.3f}")

    with open(os.path.join(args.outdir, "scores.json"), "w") as handle:
        json.dump(results, handle, indent=2)

    plot(log_power[0].detach().cpu().numpy(), maps, ground_truth, model, args)
    print(f"\nwrote {args.outdir}/scores.json and validation.png")


def plot(log_power, maps, ground_truth, model, args):
    hop = model.preproc.hop_length
    duration = log_power.shape[1] * hop / sample_rate
    extent = [0, duration, 0, sample_rate / 2000.0]
    panels = [("input log-power", log_power, "magma")] + [(k, v, "seismic") for k, v in maps.items()]
    rows = int(np.ceil(len(panels) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 3.0 * rows), constrained_layout=True)
    for ax, (title, data, cmap) in zip(np.ravel(axes), panels):
        if cmap == "seismic":
            # A diverging colormap is meaningless unless zero is pinned to the centre, and
            # a couple of outliers must not flatten everything else -- hence the percentile.
            limit = max(np.percentile(np.abs(data), 99.5), 1e-12)
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
            image = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                              extent=extent, interpolation="nearest")
        else:
            image = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, extent=extent,
                              interpolation="nearest")
        ax.contour(np.linspace(0, duration, ground_truth.shape[1]),
                   np.linspace(0, sample_rate / 2000.0, ground_truth.shape[0]),
                   ground_truth.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frequency (kHz)")
        fig.colorbar(image, ax=ax, shrink=0.85)
    for ax in np.ravel(axes)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Attribution of the mask logit inside the injected burst (green outline = ground truth)")
    fig.savefig(os.path.join(args.outdir, "validation.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()

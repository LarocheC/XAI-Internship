"""Input-domain attribution for NsNet2's suppression mask.

This replaces the approach in ``attributions_utils.compute_frequency_time_bands_attributions``,
which had three structural problems:

* it explained the *enhanced output's* band energy, which is dominated by the trivial
  pass-through path (enhanced = noisy x mask) rather than by the model's decision;
* it attributed with respect to the raw waveform, where attributions oscillate in sign at
  audio rate;
* it reduced each attribution vector to a single scalar with ``.mean()``, so the saved map
  was indexed by the *output* cell being explained and carried no input localisation at all.

Here the input is the log-power spectrogram the network actually consumes, the target is the
mean mask **logit** over a chosen time-frequency region, and the result keeps its full
``[n_freq_bins, n_frames]`` shape -- an honest answer to "which parts of the input drove this
suppression decision".
"""

import numpy as np
import torch
import torch.nn as nn
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    GradientShap,
    InputXGradient,
    IntegratedGradients,
    Saliency,
)


class RegionMaskLogit(nn.Module):
    """Maps a log-power spectrogram to the mean mask logit over a rectangular TF region.

    Captum needs a scalar-per-batch-element target; pooling over a region (rather than
    picking a single bin) also makes the explanation far less noisy.
    """

    def __init__(self, model, f_slice, t_slice):
        super().__init__()
        self.model = model
        self.f_slice = f_slice
        self.t_slice = t_slice

    def forward(self, log_power):
        logits = self.model.mask_logits(log_power)          # [B, F, T]
        region = logits[:, self.f_slice, self.t_slice]      # [B, f, t]
        return region.mean(dim=(1, 2), keepdim=False).unsqueeze(1)


def noise_floor_baselines(log_power, n_baselines=16, jitter_db=3.0, seed=0):
    """A baseline *distribution*, as DeepLiftShap and GradientShap expect.

    The original code passed ten identical copies of one tensor, a zero-variance
    "distribution" that costs ~10x the compute of a single baseline and adds no
    information. Here each baseline is the signal's own per-frequency noise floor
    (its 10th percentile over time) with a few dB of random jitter, which keeps the
    baselines on the manifold of plausible log-power spectra instead of the deeply
    saturated log(eps) corner.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    floor = torch.quantile(log_power, 0.1, dim=-1, keepdim=True)     # [B, F, 1]
    floor = floor.expand_as(log_power)
    # jitter_db is in dB of power; log-power is natural log, hence the ln(10)/10 factor
    scale = float(jitter_db) * np.log(10.0) / 10.0
    noise = torch.randn((n_baselines,) + tuple(log_power.shape[1:]),
                        generator=generator).to(log_power.device) * scale
    return floor.repeat(n_baselines, 1, 1) + noise


def attribute_region(model, log_power, f_slice, t_slice, method="deeplift",
                     baselines=None, n_steps=64, seed=0):
    """Attribution of one TF region's mean mask logit back onto the input spectrogram.

    Returns ``(attribution [F, T] as numpy, convergence_delta or None)``. The convergence
    delta is worth logging: for a completeness-based method it says how much of
    ``f(x) - f(baseline)`` the attribution actually accounts for, and it is the cheapest
    available warning that the method is not valid for the model.
    """
    target_fn = RegionMaskLogit(model, f_slice, t_slice).eval()
    inputs = log_power.clone().requires_grad_(True)

    if method == "saliency":
        return Saliency(target_fn).attribute(inputs, target=0, abs=True)[0].detach().cpu().numpy(), None
    if method == "input_x_gradient":
        return InputXGradient(target_fn).attribute(inputs, target=0)[0].detach().cpu().numpy(), None

    if baselines is None:
        baselines = noise_floor_baselines(log_power, seed=seed)

    if method == "deeplift_shap":
        # The whole point of the baseline distribution: DeepLiftShap averages over it.
        attr, delta = DeepLiftShap(target_fn).attribute(
            inputs, baselines=baselines, target=0, return_convergence_delta=True)
    elif method == "deeplift":
        attr, delta = DeepLift(target_fn).attribute(
            inputs, baselines=baselines.mean(dim=0, keepdim=True), target=0,
            return_convergence_delta=True)
    elif method == "integrated_gradients":
        attr, delta = IntegratedGradients(target_fn).attribute(
            inputs, baselines=baselines.mean(dim=0, keepdim=True), target=0, n_steps=n_steps,
            return_convergence_delta=True)
    elif method == "gradient_shap":
        attr, delta = GradientShap(target_fn).attribute(
            inputs, baselines=baselines, target=0, n_samples=32, stdevs=0.1,
            return_convergence_delta=True)
    else:
        raise ValueError(f"Unknown method: {method}")
    return attr[0].detach().cpu().numpy(), float(delta.detach().abs().mean())


def occlusion_reference(model, log_power, f_slice, t_slice, patch=(8, 4), fill=None):
    """Brute-force occlusion: the effect of each TF patch on the region's mean mask logit.

    Slow but assumption-free, so it is the natural reference to score the gradient-based
    methods against. NsNet2 is small enough that a full sweep costs seconds on CPU.
    """
    target_fn = RegionMaskLogit(model, f_slice, t_slice).eval()
    n_freq, n_frames = log_power.shape[1], log_power.shape[2]
    if fill is None:
        fill = float(torch.quantile(log_power, 0.1))
    out = np.zeros((n_freq, n_frames))
    with torch.no_grad():
        reference = float(target_fn(log_power))
        for f in range(0, n_freq, patch[0]):
            for t in range(0, n_frames, patch[1]):
                occluded = log_power.clone()
                occluded[0, f:f + patch[0], t:t + patch[1]] = fill
                out[f:f + patch[0], t:t + patch[1]] = reference - float(target_fn(occluded))
    return out


def control_maps(log_power, seed=0):
    """The two controls every attribution table needs.

    ``random`` fixes the null at exactly 1.0 enrichment by construction. ``energy`` ranks
    bins by their own log-power -- the audio analogue of Adebayo's edge detector, and a
    surprisingly strong baseline that any useful method must beat.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = tuple(log_power.shape[1:])
    return {
        "random": torch.rand(shape, generator=generator).numpy(),
        "energy_detector": log_power[0].detach().cpu().numpy() - float(log_power.min()),
    }


def localisation_scores(attribution, ground_truth_mask):
    """How much of the attribution mass lands inside a known ground-truth region.

    ``enrichment`` is the mass fraction divided by the region's area fraction, so 1.0 is
    chance. ``pointing`` is the pointing game: does the single largest-magnitude cell fall
    inside the region?
    """
    magnitude = np.abs(attribution)
    total = magnitude.sum()
    if total <= 0:
        return {"mass_fraction": 0.0, "enrichment": 0.0, "pointing": False}
    mass_fraction = float(magnitude[ground_truth_mask].sum() / total)
    area_fraction = float(ground_truth_mask.mean())
    peak = np.unravel_index(magnitude.argmax(), magnitude.shape)
    return {
        "mass_fraction": mass_fraction,
        "enrichment": mass_fraction / area_fraction if area_fraction > 0 else 0.0,
        "pointing": bool(ground_truth_mask[peak]),
    }

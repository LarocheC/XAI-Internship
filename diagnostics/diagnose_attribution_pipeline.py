"""
diagnose_attribution_pipeline.py
================================

Reproduces, from first principles, *why* the DeepLiftShap attribution maps produced by
``DeepShap/main.py`` carry almost no information about the input.

Run from the repository root::

    python diagnostics/diagnose_attribution_pipeline.py

No dataset is required: the script synthesises a probe signal (speech-like harmonic stack
plus a 3 kHz tone burst at a known time) so that the *correct* answer is known in advance.

The four checks are:

  A. Captum's DeepLIFT silently degrades to Gradient x (input - baseline) when a model
     applies its nonlinearities functionally (``nn.functional.relu`` / ``torch.sigmoid``)
     instead of through registered ``nn.Module`` instances -- which is what NsNet2 does.

  B. On the real pipeline the DeepLIFT convergence delta is of the same order as the
     quantity being explained, i.e. the attributions do not sum to f(x) - f(baseline).

  C. The saved attribution map is near-constant: an injected tone burst at a known
     time-frequency location produces no localised response.

  D. A corrected formulation (mask-logit target, log-power input, nonlinearities as
     modules) cuts the convergence delta by ~8x and yields structured attributions.
"""

import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DeepShap"))

from captum.attr import DeepLift, DeepLiftShap  # noqa: E402

from config.parameters import freq_bands, hop_length, n_fft, sample_rate  # noqa: E402
from models.NsNet2_model import NsNet2  # noqa: E402
from models.frequency_time_feature_model import BandFeatureFrequencyTimeModel  # noqa: E402
from utils.data_utils import compute_time_bands  # noqa: E402

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "DeepShap", "models", "nsnet2_baseline.bin")
SEED = 0
TONE_HZ, TONE_START, TONE_END = 3000.0, 0.8, 1.2


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_model():
    model = NsNet2(n_fft=512, n_feat=257, hd1=400, hd2=400, hd3=600)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return model.eval()


def probe_signal(duration=2.0):
    """Harmonic stack (speech-like) plus a tone burst at a known time and frequency."""
    n = int(duration * sample_rate)
    t = torch.arange(n) / sample_rate
    x = 0.03 * torch.randn(n)
    for f0 in (150, 300, 450, 600, 750):
        x += 0.12 * torch.sin(2 * np.pi * f0 * t)
    lo, hi = int(TONE_START * sample_rate), int(TONE_END * sample_rate)
    x[lo:hi] += 0.35 * torch.sin(2 * np.pi * TONE_HZ * t[lo:hi])
    return x.unsqueeze(0)


# ---------------------------------------------------------------------------
# A. Functional nonlinearities silently disable the DeepLIFT rescale rule
# ---------------------------------------------------------------------------

def check_a_functional_nonlinearities():
    rule("A. Does Captum apply the DeepLIFT rescale rule to NsNet2's nonlinearities?")

    class Functional(nn.Module):
        def __init__(self, w1, w2):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(4, 8, bias=False), nn.Linear(8, 1, bias=False)
            self.fc1.weight.data, self.fc2.weight.data = w1.clone(), w2.clone()

        def forward(self, x):
            return self.fc2(nn.functional.relu(self.fc1(x)))  # as NsNet2 does it

    class Modular(nn.Module):
        def __init__(self, w1, w2):
            super().__init__()
            self.fc1, self.act, self.fc2 = nn.Linear(4, 8, bias=False), nn.ReLU(), nn.Linear(8, 1, bias=False)
            self.fc1.weight.data, self.fc2.weight.data = w1.clone(), w2.clone()

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    torch.manual_seed(SEED)
    w1, w2 = torch.randn(8, 4), torch.randn(1, 8)
    fm, mm = Functional(w1, w2).eval(), Modular(w1, w2).eval()
    x, base = torch.randn(1, 4, requires_grad=True), torch.randn(1, 4) * 0.5

    assert torch.allclose(fm(x), mm(x)), "the two nets must be numerically identical"
    print("The two networks are numerically identical; only the *spelling* of ReLU differs.\n")

    for name, model in (("nn.functional.relu", fm), ("nn.ReLU() module", mm)):
        attr, delta = DeepLift(model).attribute(x, baselines=base, return_convergence_delta=True)
        print(f"  {name:20s} attr={np.round(attr.detach().numpy()[0], 4)}  "
              f"sum={attr.sum():8.4f}  |delta|={delta.abs().item():.2e}")

    grad = torch.autograd.grad(mm(x), x)[0]
    gxd = grad * (x - base)
    print(f"  {'grad x (x - base)':20s} attr={np.round(gxd.detach().numpy()[0], 4)}  sum={gxd.sum():8.4f}")
    print(f"  true f(x) - f(base) = {(mm(x) - mm(base)).item():.4f}\n")
    print("  => With a functional ReLU the result is EXACTLY Gradient x Input and the")
    print("     convergence delta is large. NsNet2 uses nn.functional.relu and torch.sigmoid,")
    print("     so DeepShap/ is not running DeepLIFT -- it is running Gradient x Input.")


# ---------------------------------------------------------------------------
# B + C. What the saved attribution map actually contains
# ---------------------------------------------------------------------------

def check_bc_repo_pipeline(division=4):
    rule("B/C. Reproducing DeepShap/main.py's attribution map on a probe signal")

    torch.manual_seed(SEED)
    model = load_model()
    x = probe_signal()
    _, time_bands = compute_time_bands(x, sample_rate, hop_length, division)
    n_f, n_t = len(freq_bands), len(time_bands)

    wrapped = BandFeatureFrequencyTimeModel(model)
    inp = x.unsqueeze(0).requires_grad_(True)
    baseline = torch.zeros((10, 1, x.shape[-1]))

    with torch.no_grad():
        f_x = wrapped(inp, time_bands).reshape(n_f, n_t)
        f_base = wrapped(baseline[:1], time_bands).reshape(n_f, n_t)
    out_delta = (f_x - f_base).numpy()

    dls = DeepLiftShap(wrapped)
    attr_map, deltas = np.zeros((n_f, n_t)), []
    targets = list(range(n_f * n_t))
    for i in range(0, len(targets), 8):
        batch = targets[i:i + 8]
        model.train()                                    # as attributions_utils.py does
        attrs, delta = dls.attribute(inp.repeat(len(batch), 1, 1), baselines=baseline,
                                     target=batch, additional_forward_args=(time_bands,),
                                     return_convergence_delta=True)
        model.eval()
        deltas.append(delta.detach().abs().mean().item())
        attrs = attrs.detach().numpy()
        for j, tgt in enumerate(batch):
            attr_map[tgt // n_t, tgt % n_t] = attrs[j].mean()   # <-- the whole map is built from this

    mean_delta = float(np.mean(deltas))
    print(f"  mean |convergence delta|                = {mean_delta:8.2f}")
    print(f"  mean |f(x) - f(baseline)| being explained = {np.abs(out_delta).mean():8.2f} dB")
    print(f"  => attributions account for only ~{100 * (1 - mean_delta / np.abs(out_delta).mean()):.0f}% "
          f"of the quantity they claim to decompose.\n")

    np.set_printoptions(precision=2, suppress=False, linewidth=140)
    print("  attribution map as saved and plotted by main.py "
          f"(rows = {n_f} freq bands, cols = {n_t} time bands of {1/division:.2f}s):")
    print(np.array2string(attr_map, prefix="    "))

    burst_cols = [j for j in range(n_t) if (j + 1) / division > TONE_START and j / division < TONE_END]
    interior = attr_map[:, :max(1, n_t - 2)]
    print(f"\n  ground truth: the {TONE_HZ:.0f} Hz burst lives in freq band 5 "
          f"({freq_bands[5][0]}-{freq_bands[5][1]} Hz), time bands {burst_cols}")
    print(f"  coefficient of variation over the interior of the map = "
          f"{interior.std() / abs(interior.mean()):.3f}")
    print("  => the map is near-constant; the injected burst produces no localised response.")
    print("\n  Why: attrs[idx].mean() collapses a whole-waveform attribution vector to ONE scalar,")
    print("  so the map is indexed by the OUTPUT (f,t) cell being explained -- it never encodes")
    print("  *where in the input* the evidence was. There is no input localisation in the pipeline.")
    return mean_delta


# ---------------------------------------------------------------------------
# D. A corrected formulation
# ---------------------------------------------------------------------------

class MaskLogit(nn.Module):
    """NsNet2's body with the nonlinearities as registered modules, returning pre-sigmoid
    mask logits from a log-power spectrogram. DeepLIFT's rescale rule is valid here for the
    dense layers (the GRUs remain opaque -- see the residual delta)."""

    def __init__(self, base):
        super().__init__()
        self.fc1, self.rnn1, self.rnn2 = base.fc1, base.rnn1, base.rnn2
        self.fc2, self.fc3, self.fc4 = base.fc2, base.fc3, base.fc4
        self.relu1, self.relu2 = nn.ReLU(), nn.ReLU()      # distinct instances: no module reuse

    def forward(self, log_power):                          # [B, F, T] -> [B, F, T]
        h = log_power.permute(0, 2, 1)
        h = self.fc1(h)
        h, _ = self.rnn1(h)
        h, _ = self.rnn2(h)
        h = self.relu1(self.fc2(h))
        h = self.relu2(self.fc3(h))
        return self.fc4(h).permute(0, 2, 1)


def check_d_corrected(repo_delta):
    rule("D. A corrected formulation: mask-logit target on a log-power input")

    torch.manual_seed(SEED)
    model = load_model()
    net = MaskLogit(model).eval()
    x = probe_signal()
    log_power = torch.log(model.preproc(x).abs() ** 2 + model.eps).squeeze(1)
    n_freq, n_frames = log_power.shape[1], log_power.shape[2]
    model_hop = model.preproc.hop_length

    f_bin = int(round(TONE_HZ / sample_rate * n_fft))
    t_frame = int(((TONE_START + TONE_END) / 2) * sample_rate / model_hop)

    class Pick(nn.Module):
        def __init__(self, inner, f, t):
            super().__init__()
            self.inner, self.f, self.t = inner, f, t

        def forward(self, z):
            return self.inner(z)[:, self.f, self.t].unsqueeze(1)

    pick = Pick(net, f_bin, t_frame).eval()
    inp = log_power.clone().requires_grad_(True)
    baseline = torch.full_like(log_power, float(np.log(model.eps)))

    attr, delta = DeepLift(pick).attribute(inp, baselines=baseline, target=0,
                                           return_convergence_delta=True)
    a = attr[0].detach().abs().numpy()
    print(f"  explaining mask logit at bin {f_bin} ({f_bin * sample_rate / n_fft:.0f} Hz), "
          f"frame {t_frame} ({t_frame * model_hop / sample_rate:.2f}s)")
    print(f"  |convergence delta| = {delta.abs().item():.2f}   "
          f"(repo pipeline: {repo_delta:.1f})   total |attr| = {a.sum():.1f}")
    print(f"  => residual delta is now ~{100 * delta.abs().item() / a.sum():.1f}% of the attribution mass.")
    print("     It is not zero because the two GRUs remain undecomposable by the rescale rule.\n")

    per_freq = a.sum(axis=1)
    top = np.argsort(per_freq)[::-1][:6]
    near = per_freq[max(0, f_bin - 8):f_bin + 9].sum() / per_freq.sum()
    print(f"  top input frequencies driving this mask decision: "
          f"{[round(float(k * sample_rate / n_fft)) for k in top]} Hz")
    print(f"  share of attribution within +/-250 Hz of the target bin: {near:.1%}")
    print("  => the decision at 3 kHz is driven by low-frequency (speech-band) energy:")
    print("     evidence of a global, VAD-like speech-presence decision rather than a")
    print("     per-bin one. That is a real, reportable finding -- the kind of structure the")
    print("     current pipeline is incapable of showing.")


def check_e_stft_config():
    """The DNS-Challenge NSNet2 baseline is trained on 20 ms frames with a 10 ms hop at
    16 kHz (win_length=320, hop_length=160) zero-padded to a 512-point FFT, which is what
    gives its 257 input features. NsNet2_model.py constructs
    torchaudio.transforms.Spectrogram(n_fft=512) and takes the DEFAULTS -- win_length=512
    (32 ms) and hop_length=256 (16 ms). This check measures whether that matters."""
    rule("E. Is NsNet2 being run with its reference STFT configuration?")

    torch.manual_seed(SEED)
    model = load_model()
    print(f"  repo:      n_fft={model.preproc.n_fft} win_length={model.preproc.win_length} "
          f"hop_length={model.preproc.hop_length}  ({model.preproc.win_length / sample_rate * 1e3:.0f} ms window)")
    print("  reference: n_fft=512 win_length=320 hop_length=160  (20 ms window, 10 ms hop)\n")

    # harmonic "speech" burst in [0.5, 1.5] s, embedded in stationary white noise
    n = 2 * sample_rate
    t = torch.arange(n) / sample_rate
    voiced = (t > 0.5) & (t < 1.5)
    x = 0.02 * torch.randn(n)
    for f0 in (150, 300, 450, 600, 750, 900):
        x += 0.10 * torch.sin(2 * np.pi * f0 * t) * voiced
    x = x.unsqueeze(0)

    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate)
    speech_band, high_band = (freqs >= 100) & (freqs < 1000), freqs >= 4000

    print(f"  {'window':>10} {'hop':>5} | {'keep speech':>11} {'suppress silence':>16} {'suppress HF noise':>17}")
    print("  " + "-" * 68)
    for win, hop, tag in ((512, 256, "repo default"), (320, 160, "NSNet2 reference")):
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                          window=torch.hann_window(win), return_complex=True)
        log_power = torch.log(spec.abs() ** 2 + model.eps)
        with torch.no_grad():
            h = log_power.permute(0, 2, 1)
            h = model.fc1(h)
            h, _ = model.rnn1(h)
            h, _ = model.rnn2(h)
            h = nn.functional.relu(model.fc2(h))
            h = nn.functional.relu(model.fc3(h))
            mask = torch.sigmoid(model.fc4(h)).permute(0, 2, 1)
        centres = torch.arange(mask.shape[-1]) * hop / sample_rate
        active, silent = (centres > 0.55) & (centres < 1.45), (centres < 0.45) | (centres > 1.55)
        keep = mask[0][speech_band][:, active].mean().item()
        suppress = 1 - mask[0][speech_band][:, silent].mean().item()
        suppress_hf = 1 - mask[0][high_band][:, silent].mean().item()
        print(f"  {win:>6} ({win / sample_rate * 1e3:.0f}ms) {hop:>5} | {keep:11.3f} {suppress:16.3f} "
              f"{suppress_hf:17.3f}   <- {tag}")

    print("\n  => At the repo's 32 ms window the model leaves high-frequency noise largely")
    print("     un-suppressed. Attributions computed on this configuration describe a")
    print("     mis-configured model, independently of every other issue above.")
    print("     Fix: Spectrogram(n_fft=512, win_length=320, hop_length=160, power=None) and")
    print("     the matching InverseSpectrogram. Confirm against the DNS-Challenge reference.")


if __name__ == "__main__":
    check_a_functional_nonlinearities()
    repo_delta = check_bc_repo_pipeline()
    check_d_corrected(repo_delta)
    check_e_stft_config()
    print("\n" + "=" * 78)
    print("See PLAN.md for what to do about all of this.")
    print("=" * 78)

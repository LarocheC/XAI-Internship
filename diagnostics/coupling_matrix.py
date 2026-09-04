"""Verify the flagship claim on REAL speech: is NsNet2's mask decision a rank-one global gate?

R[f_in, f_out] = sum_t |d z(t0, f_out) / d L(t, f_in)|   (257 backward passes on one graph)
Control: an oracle per-bin Wiener logit, diagonal by construction -- the apparatus must
detect locality there, or the metric is meaningless.
"""
import sys, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/user/XAI-Internship/DeepShap")
torch.manual_seed(0); torch.set_num_threads(4)
from utils.model_utils import load_nsnet2_model
from validate_attributions import read_wav, synthetic_speech

m,_=load_nsnet2_model(); m.eval()
SR=16000
import argparse
_ap=argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--input", default=None, help="16 kHz 16-bit mono wav; synthesised if omitted")
_args=_ap.parse_args()
clean=read_wav(_args.input, max_seconds=3.0) if _args.input else synthetic_speech(3.0)
clean=clean/clean.abs().max()*0.5
if _args.input is None:
    print("NOTE: no --input given, using synthetic speech. The VAD may not engage on it;\n"
          "      check the speech/pause gain ratio below before trusting the numbers.")
g=torch.Generator().manual_seed(0)
noise=torch.randn(1,clean.shape[-1],generator=g); noise=noise/noise.std()*clean.std()/10**(5/20)
noisy=clean+noise

lp=m.log_power(m.preproc(noisy)); F_,T_=lp.shape[1],lp.shape[2]
with torch.no_grad():
    z=m.mask_logits(lp)[0]
    gain=torch.sigmoid(z)
# real-speech sanity: is the VAD actually firing?
cl=m.log_power(m.preproc(clean))[0]
frame_energy=cl.mean(0); thr=frame_energy.quantile(0.6)
sp=frame_energy>thr; pa=frame_energy<frame_energy.quantile(0.25)
print(f"REAL-SPEECH SANITY  mean mask gain: speech frames {gain[:,sp].mean():.3f}  pause frames {gain[:,pa].mean():.3f}"
      f"  ratio {(gain[:,sp].mean()/gain[:,pa].mean()):.2f}x")
print(f"saturation: {(z.abs()>4.6).float().mean()*100:.1f}% of logits with |z|>4.6, median |z|={z.abs().median():.2f}\n")

def coupling(fn, t0):
    x=lp.clone().requires_grad_(True)
    out=fn(x)[0]                       # [F,T]
    R=np.zeros((F_,F_))
    for f_out in range(F_):
        gr,=torch.autograd.grad(out[f_out,t0], x, retain_graph=True)
        R[:,f_out]=gr[0].abs().sum(dim=1).numpy()
    return R

def stats(R,band=8):
    s=np.linalg.svd(R,compute_uv=False); s2=s**2
    top1=s2[0]/s2.sum(); pr=(s2.sum()**2)/(s2**2).sum()
    ii,jj=np.indices(R.shape); near=np.abs(ii-jj)<=band
    enr=(R[near].sum()/R.sum())/(near.sum()/R.size)
    return top1,pr,enr

# oracle Wiener logit: per-bin, diagonal by construction
def wiener(x):
    snr=torch.exp(x)/ (torch.exp(x).mean(dim=-1,keepdim=True)*0.3+1e-8)
    return torch.log(snr/(1+snr)+1e-8).unsqueeze(0) if x.dim()==2 else torch.log(snr/(1+snr)+1e-8)

frames=[int(T_*0.35), int(T_*0.55), int(T_*0.75)]
print(f"{'system':>16} {'frame':>6} {'top-1 SV':>9} {'part.ratio':>11} {'diag enrich':>12}")
for name,fn in [("NsNet2 mask", lambda x: m.mask_logits(x)), ("oracle Wiener", wiener)]:
    for t0 in frames:
        R=coupling(fn,t0); a,b,c=stats(R)
        print(f"{name:>16} {t0:6d} {a:9.3f} {b:11.2f} {c:12.2f}")

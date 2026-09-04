"""Causal test: if the gain surface is really rank ~5, truncating it there should cost nothing.

Control: a RANDOM rank-r surface with the same energy, to show the specific subspace matters
and it is not just 'any smooth gain works'.
"""
import sys, warnings, glob, numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/user/XAI-Internship/DeepShap")
torch.manual_seed(0); np.random.seed(0)
import soundfile as sf
from utils.model_utils import load_nsnet2_model
SR=16000; S="/tmp/claude-0/-home-user-XAI-Internship/d0cc0c4e-e629-5c87-9c02-44b0520c41b3/scratchpad"
m,_=load_nsnet2_model(); m.eval()

def si_sdr(est,ref):
    est=est-est.mean(); ref=ref-ref.mean()
    a=(est*ref).sum()/(ref*ref).sum(); e=est-a*ref
    return float(10*torch.log10((a*ref).pow(2).sum()/(e.pow(2).sum()+1e-12)+1e-12))

def lsd(est,ref):  # log-spectral distance, dB
    E=torch.stft(est,512,256,window=torch.hann_window(512),return_complex=True).abs()+1e-8
    R=torch.stft(ref,512,256,window=torch.hann_window(512),return_complex=True).abs()+1e-8
    n=min(E.shape[-1],R.shape[-1])
    return float(((20*torch.log10(E[...,:n]/R[...,:n]))**2).mean().sqrt())

def apply_gain(G, stft, length):
    return torch.istft(stft*G.unsqueeze(0), 512, 256, window=torch.hann_window(512), length=length)

files=sorted(glob.glob(f"{S}/LibriSpeech/test-clean/*/*/*.flac"))[:6]
ranks=[1,2,4,8,16,32]
acc={f"r{r}":[] for r in ranks}; acc["full"]=[]; acc["noisy"]=[]
acc.update({f"rand{r}":[] for r in [4,16]})
for fp in files:
    a,_=sf.read(fp,dtype="float32"); a=torch.tensor(a[:int(3*SR)]); a=a/a.abs().max()*0.5
    clean=a.unsqueeze(0); L=clean.shape[-1]
    g=torch.Generator().manual_seed(1); n=torch.randn(clean.shape,generator=g)
    n=n/n.std()*clean.std()/10**(5/20); noisy=clean+n
    with torch.no_grad():
        stft=m.preproc(noisy).squeeze(1)[0]
        G=torch.sigmoid(m.mask_logits(m.log_power(m.preproc(noisy))))[0]
    T=min(stft.shape[-1],G.shape[-1]); stft,G=stft[:,:T],G[:,:T]
    ref=clean[0][:L]
    acc["noisy"].append((si_sdr(noisy[0],ref), lsd(noisy[0],ref)))
    full=apply_gain(G,stft,L); acc["full"].append((si_sdr(full,ref), lsd(full,ref)))
    U,Sv,Vh=torch.linalg.svd(G-G.mean(), full_matrices=False)
    for r in ranks:
        Gr=(U[:,:r]*Sv[:r])@Vh[:r]+G.mean()
        y=apply_gain(Gr.clamp(0,1),stft,L); acc[f"r{r}"].append((si_sdr(y,ref), lsd(y,ref)))
    for r in [4,16]:                              # control: random rank-r, matched energy
        gg=torch.Generator().manual_seed(7)
        A=torch.randn(G.shape[0],r,generator=gg); B=torch.randn(r,G.shape[1],generator=gg)
        Gr=A@B; Gr=(Gr-Gr.mean())/Gr.std()*(G-G.mean()).std()+G.mean()
        y=apply_gain(Gr.clamp(0,1),stft,L); acc[f"rand{r}"].append((si_sdr(y,ref), lsd(y,ref)))
print(f"{'gain surface':>22} {'SI-SDR dB':>10} {'LSD dB':>8}")
for k in ["noisy","full"]+[f"r{r}" for r in ranks]+["rand4","rand16"]:
    v=np.array(acc[k]); lbl = "  <- control (random subspace)" if k.startswith("rand") else ""
    print(f"{k:>22} {v[:,0].mean():10.2f} {v[:,1].mean():8.2f}{lbl}")
print(f"\n(mean over {len(files)} LibriSpeech utterances at 5 dB SNR)")

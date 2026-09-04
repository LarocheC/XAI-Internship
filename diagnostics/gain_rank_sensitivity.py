"""Does r90 move AT ALL on the eco8 NSNet2 family?

If it is pinned at 2.0 regardless of SNR, noise type and utterance, the family sweep
measured nothing and no reading may be taken from it.
"""
import sys, json, glob, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/user/larochec/eco8-neaixt")
torch.manual_seed(0); np.random.seed(0)
import soundfile as sf
from nsnet2.model import NSNet2
SR=16000; S="/tmp/claude-0/-home-user-XAI-Internship/d0cc0c4e-e629-5c87-9c02-44b0520c41b3/scratchpad"
class H:
    def __init__(s,d): s.__dict__.update(d)
def er(G,frac=0.90):
    sv=np.linalg.svd(G-G.mean(),compute_uv=False); c=np.cumsum(sv**2)/(sv**2).sum()
    return int(np.searchsorted(c,frac)+1)
cfg=json.load(open(f"{S}/ns2/baseline/config.json")); m=NSNet2(H(cfg))
sd=torch.load(f"{S}/ns2/baseline/g_best",map_location="cpu",weights_only=True)
m.load_state_dict(sd.get("generator",sd) if isinstance(sd,dict) else sd,strict=False); m.eval()
cf=float(cfg.get("compress_factor",0.3))
def gain(noisy):
    sp=torch.stft(noisy,512,256,window=torch.hann_window(512),return_complex=True)
    mag=(sp.abs()**cf).unsqueeze(0)
    with torch.no_grad():
        x=mag.transpose(1,2); h=m.act(m.fc_in(x)); h,_=m.gru(h)
        h=m.act(m.fc1(h)); h=m.act(m.fc2(h))
        return torch.sigmoid(m.fc_out(h)).transpose(1,2)[0].numpy()
def mk(kind,n,seed):
    g=torch.Generator().manual_seed(seed)
    if kind=="white": return torch.randn(n,generator=g)
    if kind=="pink":
        W=torch.fft.rfft(torch.randn(n,generator=g)); f=torch.arange(W.shape[-1]).float().clamp(min=1)
        return torch.fft.irfft(W/f.sqrt(),n=n)
    if kind=="tone":
        t=torch.arange(n)/SR; return sum(torch.sin(2*np.pi*f0*t) for f0 in (440,880,1760))
    if kind=="hf":
        W=torch.fft.rfft(torch.randn(n,generator=g)); f=torch.arange(W.shape[-1]).float()
        return torch.fft.irfft(W*(f>f.max()*0.5),n=n)
files=sorted(glob.glob(f"{S}/LibriSpeech/test-clean/*/*/*.flac"))[:5]
print("r90 / r99 of the gain surface, eco8 NSNet2 baseline\n")
print(f"{'condition':>22} " + " ".join(f"{u:>12}" for u in ["utt1","utt2","utt3","utt4","utt5"]))
allv=[]
for kind in ["white","pink","tone","hf"]:
    for snr in [-5,5,20]:
        row=[]
        for fp in files:
            a,_=sf.read(fp,dtype="float32"); a=torch.tensor(a[:int(3*SR)]); a=a/a.abs().max()*0.5
            n=mk(kind,a.shape[0],7); n=n/n.std()*a.std()/10**(snr/20)
            G=gain(a+n); row.append((er(G),er(G,.99))); allv.append(er(G))
        print(f"{kind+' @ '+str(snr)+'dB':>22} " + " ".join(f"{r[0]:>5}/{r[1]:<6}" for r in row))
print(f"\nr90 across all {len(allv)} conditions: min={min(allv)} max={max(allv)} "
      f"unique={sorted(set(allv))}")
print("If r90 never leaves 2, the family sweep measured nothing.")

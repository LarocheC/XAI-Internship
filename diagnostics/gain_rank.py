"""Forward-only alternative: effective rank of the PREDICTED GAIN SURFACE G[f,t]=sigma(z).

No Jacobian, no abs-before-sum. Reported as r90/r99 (number of singular values holding
90%/99% of spectral energy), which unlike top-1 SV has a usable null and ceiling.
Nulls and ceilings measured through the identical code:
  random gain          -> the null (what a structureless surface scores)
  oracle IRM / Wiener  -> the ceiling (what the training target itself scores)
  phase-shuffled G     -> destroys T-F structure, keeps the marginal distribution
"""
import sys, warnings, glob, numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/user/XAI-Internship/DeepShap")
torch.manual_seed(0); np.random.seed(0)
import soundfile as sf
from utils.model_utils import load_nsnet2_model
SR=16000; S="/tmp/claude-0/-home-user-XAI-Internship/d0cc0c4e-e629-5c87-9c02-44b0520c41b3/scratchpad"
m,_=load_nsnet2_model(); m.eval()

def er(G, frac):
    s=np.linalg.svd(G-G.mean(), compute_uv=False)   # centred: a constant surface is rank 0 of *structure*
    c=np.cumsum(s**2)/ (s**2).sum()
    return int(np.searchsorted(c, frac)+1)

files=sorted(glob.glob(f"{S}/LibriSpeech/test-clean/*/*/*.flac"))[:6]
print(f"{'system':>24} " + " ".join(f"{s:>10}" for s in ["r90@0dB","r90@5dB","r90@15dB","r99@5dB"]))
rows={k:[] for k in ["NSNet2 gain","oracle IRM","oracle Wiener","random gain","shuffled NSNet2"]}
for fp in files:
    a,sr=sf.read(fp,dtype="float32"); a=torch.tensor(a[:int(3*SR)]); a=a/a.abs().max()*0.5
    clean=a.unsqueeze(0)
    for snr in [0,5,15]:
        g=torch.Generator().manual_seed(1)
        n=torch.randn(clean.shape,generator=g); n=n/n.std()*clean.std()/10**(snr/20)
        noisy=clean+n
        with torch.no_grad():
            G=torch.sigmoid(m.mask_logits(m.log_power(m.preproc(noisy))))[0].numpy()
            C=m.preproc(clean).abs().squeeze(1)[0].numpy(); N=m.preproc(n).abs().squeeze(1)[0].numpy()
        T=min(G.shape[1],C.shape[1]); G,C,N=G[:,:T],C[:,:T],N[:,:T]
        irm=C/(C+N+1e-9); wien=C**2/(C**2+N**2+1e-9)
        rnd=np.random.rand(*G.shape)
        sh=G.copy(); idx=np.random.permutation(sh.size); sh=sh.flatten()[idx].reshape(G.shape)
        for k,M in [("NSNet2 gain",G),("oracle IRM",irm),("oracle Wiener",wien),
                    ("random gain",rnd),("shuffled NSNet2",sh)]:
            rows[k].append((snr, er(M,0.90), er(M,0.99)))
for k,v in rows.items():
    r=lambda s,i: np.mean([x[i] for x in v if x[0]==s])
    print(f"{k:>24} {r(0,1):10.1f} {r(5,1):10.1f} {r(15,1):10.1f} {r(5,2):10.1f}")
print(f"\n(mean over {len(files)} LibriSpeech utterances, 257 x ~187 gain surfaces)")

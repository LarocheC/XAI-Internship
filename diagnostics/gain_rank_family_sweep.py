"""Within-family sweep: does gain-surface effective rank track parameter redundancy?

Architecture is held FIXED (eco8 NSNet2); only width and factorisation change across the
12 public checkpoints. That is the design the failed cross-architecture test could not give:
no input-representation confound, no differing oracle, no differing training recipe.
"""
import sys, os, json, glob, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/larochec/eco8-neaixt")
torch.manual_seed(0); np.random.seed(0)
import soundfile as sf
from nsnet2.model import NSNet2
SR=16000; S="/tmp/claude-0/-home-user-XAI-Internship/d0cc0c4e-e629-5c87-9c02-44b0520c41b3/scratchpad"

class H:
    def __init__(s,d): s.__dict__.update(d)

def er(G,frac=0.90):
    sv=np.linalg.svd(G-G.mean(),compute_uv=False); c=np.cumsum(sv**2)/(sv**2).sum()
    return int(np.searchsorted(c,frac)+1)

def load(sub):
    cfg=json.load(open(f"{S}/ns2/{sub}/config.json"))
    m=NSNet2(H(cfg))
    sd=torch.load(f"{S}/ns2/{sub}/g_best",map_location="cpu",weights_only=True)
    sd=sd.get("generator",sd) if isinstance(sd,dict) else sd
    m.load_state_dict(sd,strict=False); m.eval()
    return m, float(cfg.get("compress_factor",0.3)), sum(p.numel() for p in m.parameters())

files=sorted(glob.glob(f"{S}/LibriSpeech/test-clean/*/*/*.flac"))[:6]
subs=sorted([os.path.basename(d) for d in glob.glob(f"{S}/ns2/*") if os.path.isdir(d) and not d.endswith('.cache')])
print(f"{'variant':>16} {'params':>10} {'r90':>7} {'r99':>7} {'IRM r90':>8}")
print("-"*56)
res=[]
for sub in subs:
    try: m,cf,P = load(sub)
    except Exception as e:
        print(f"{sub:>16}   SKIP {type(e).__name__}: {str(e)[:40]}"); continue
    r90=[];r99=[];i90=[]
    for fp in files:
        a,_=sf.read(fp,dtype="float32"); a=torch.tensor(a[:int(3*SR)]); a=a/a.abs().max()*0.5
        g=torch.Generator().manual_seed(1); n=torch.randn(a.shape,generator=g)
        n=n/n.std()*a.std()/10**(5/20); noisy=a+n
        sp=torch.stft(noisy,512,256,window=torch.hann_window(512),return_complex=True)
        mag=(sp.abs()**cf).unsqueeze(0); pha=sp.angle().unsqueeze(0)
        with torch.no_grad():
            x=mag.transpose(1,2); h=m.act(m.fc_in(x)); h,_=m.gru(h)
            h=m.act(m.fc1(h)); h=m.act(m.fc2(h))
            G=torch.sigmoid(m.fc_out(h)).transpose(1,2)[0].numpy()
        C=torch.stft(a,512,256,window=torch.hann_window(512),return_complex=True).abs().numpy()
        N=torch.stft(n,512,256,window=torch.hann_window(512),return_complex=True).abs().numpy()
        T=min(G.shape[1],C.shape[1]); G,C,N=G[:,:T],C[:,:T],N[:,:T]
        irm=(C**cf)/((C**cf)+(N**cf)+1e-9)
        r90.append(er(G)); r99.append(er(G,.99)); i90.append(er(irm))
    print(f"{sub:>16} {P:10,} {np.mean(r90):7.1f} {np.mean(r99):7.1f} {np.mean(i90):8.1f}")
    res.append((sub,P,np.mean(r90),np.mean(r99)))
if len(res)>3:
    from scipy.stats import spearmanr
    P=np.array([r[1] for r in res]); R=np.array([r[2] for r in res]); R9=np.array([r[3] for r in res])
    print(f"\nSpearman(params, r90) = {spearmanr(P,R).statistic:+.3f}   "
          f"Spearman(params, r99) = {spearmanr(P,R9).statistic:+.3f}   (n={len(res)})")
    print("HYPOTHESIS: more parameters -> more redundancy -> LOWER gain-surface rank => expect NEGATIVE")

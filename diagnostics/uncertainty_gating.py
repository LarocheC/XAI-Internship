"""v2: per-BIN gating, and a testbed where the enhancer actually damages speech.

Two fixes to v1:
  * risk is predicted PER TIME-FREQUENCY BIN, not per frame. v1 collapsed it to a frame
    gate, so a flagged frame had its whole spectrum un-suppressed -- which is why v1's
    'oracle' scored BELOW the probe and was therefore no ceiling at all.
  * the test set is REVERBERANT (the shift under which the eco8/dnsmos study found a
    global mask-softening knob worth +0.64 PESQ). On matched conditions v1 showed there
    is no damage to prevent, so no method can win.
The probe is trained on ANECHOIC audio only: it must generalise to the shift unseen.
"""
import sys, glob, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/user/XAI-Internship/DeepShap")
torch.manual_seed(0); np.random.seed(0)
import soundfile as sf
from pesq import pesq as pesq_fn
from pystoi import stoi as stoi_fn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils.model_utils import load_nsnet2_model
SR=16000; S="/tmp/claude-0/-home-user-XAI-Internship/d0cc0c4e-e629-5c87-9c02-44b0520c41b3/scratchpad"
m,_=load_nsnet2_model(); m.eval(); HOP=m.preproc.hop_length

def si_sdr(e,r):
    e=e-e.mean(); r=r-r.mean(); a=(e*r).sum()/(r*r).sum(); z=e-a*r
    return float(10*torch.log10((a*r).pow(2).sum()/(z.pow(2).sum()+1e-12)+1e-12))
def mk_noise(kind,n,seed):
    g=torch.Generator().manual_seed(seed)
    if kind=="white": return torch.randn(n,generator=g)
    if kind=="pink":
        W=torch.fft.rfft(torch.randn(n,generator=g)); f=torch.arange(W.shape[-1]).float().clamp(min=1)
        return torch.fft.irfft(W/f.sqrt(),n=n)
    if kind=="babble":
        b=torch.zeros(n)
        for k in range(1,7): b+=torch.roll(torch.randn(n,generator=g),k*977)
        W=torch.fft.rfft(b); f=torch.arange(W.shape[-1]).float().clamp(min=1)
        return torch.fft.irfft(W/(f**0.7),n=n)
    if kind=="hf":
        W=torch.fft.rfft(torch.randn(n,generator=g)); f=torch.arange(W.shape[-1]).float()
        return torch.fft.irfft(W*(f>f.max()*0.45),n=n)
def reverb(x,rt60=0.45,seed=0):
    g=torch.Generator().manual_seed(seed); L=int(0.6*SR)
    t=torch.arange(L)/SR
    h=torch.randn(L,generator=g)*torch.exp(-6.9*t/rt60); h[0]=1.0; h=h/h.norm()
    y=torch.nn.functional.conv1d(x.view(1,1,-1),h.flip(0).view(1,1,-1),padding=L-1)[0,0,:x.shape[0]]
    return y/y.abs().max()*x.abs().max()

def analyse(clean,noise):
    noisy=clean+noise
    lp=m.log_power(m.preproc(noisy.unsqueeze(0)))
    h=lp.permute(0,2,1); h=m.fc1(h); g1,_=m.rnn1(h); g2,_=m.rnn2(g1)
    z=m.fc4(m.relu2(m.fc3(m.relu1(m.fc2(g2))))).permute(0,2,1)
    G=torch.sigmoid(z)[0]
    X=m.preproc(noisy.unsqueeze(0)).squeeze(1)[0]
    C=m.preproc(clean.unsqueeze(0)).abs().squeeze(1)[0]
    N=m.preproc(noise.unsqueeze(0)).abs().squeeze(1)[0]
    T=min(G.shape[1],X.shape[1],C.shape[1],N.shape[1])
    G,X,C,N=G[:,:T],X[:,:T],C[:,:T],N[:,:T]
    ibm=(C>N).float()
    risk=(ibm*(1-G))                                  # per-BIN over-suppression
    # per-bin features: internal state (400) broadcast + this bin's own gain and log-power
    st=g2[0][:T].detach().numpy()                     # [T,400]
    return G.detach(),X,C,N,ibm,risk.detach(),st,lp[0][:,:T].detach().numpy()

def binfeats(st,G,lpm,fsub):
    """[T*|fsub|, D] per-bin features: frame state (compressed) + local gain + local level."""
    T=st.shape[0]
    proj=st[:,:64]                                     # cheap 64-d slice of the 400-d state
    out=[]
    for f in fsub:
        out.append(np.concatenate([proj,
                                   G[f,:T].numpy()[:,None],
                                   lpm[f,:T][:,None],
                                   np.full((T,1),f/257.0)],axis=1))
    return np.concatenate(out,axis=0)

def synth(G,X,L): return torch.istft(X.unsqueeze(0)*G.unsqueeze(0),512,HOP,
                                     window=torch.hann_window(512),length=L)[0]
def score(e,r):
    a=e.numpy(); b=r.numpy(); n=min(len(a),len(b))
    try: p=pesq_fn(SR,b[:n],a[:n],'wb')
    except Exception: p=float('nan')
    try: s=stoi_fn(b[:n],a[:n],SR,extended=True)
    except Exception: s=float('nan')
    return p,s,si_sdr(e[:n],r[:n])

files=sorted(glob.glob(f"{S}/LibriSpeech/test-clean/*/*/*.flac"))
train_f,test_f=files[:20],files[40:64]
KINDS=["white","pink","babble","hf"]; SNRS=[0,5,10]
FSUB=np.arange(4,256,6)                                # 42 frequency bins, evenly spread

# ---- train per-bin risk probe on ANECHOIC audio only ----
Xtr=[];ytr=[]
for i,fp in enumerate(train_f):
    a,_=sf.read(fp,dtype="float32"); a=torch.tensor(a[:int(3*SR)]); a=a/a.abs().max()*0.5
    for k in KINDS:
        for snr in SNRS:
            nz=mk_noise(k,a.shape[0],hash((i,k,snr))%9999); nz=nz/nz.std()*a.std()/10**(snr/20)
            G,X,C,N,ibm,risk,st,lpm=analyse(a,nz)
            Xtr.append(binfeats(st,G,lpm,FSUB))
            ytr.append(np.concatenate([(risk[f,:st.shape[0]].numpy()>0.5).astype(float) for f in FSUB]))
Xtr=np.concatenate(Xtr); ytr=np.concatenate(ytr)
sc=StandardScaler().fit(Xtr); clf=LogisticRegression(max_iter=1200,C=0.1).fit(sc.transform(Xtr),ytr)
print(f"per-bin risk probe: {len(ytr):,} samples ({ytr.mean()*100:.1f}% positive), "
      f"{Xtr.shape[1]}-d  [trained ANECHOIC only]\n")

def full_risk(st,G,lpm):
    T=st.shape[0]; U=np.zeros((257,T))
    allf=np.arange(257)
    F=binfeats(st,G,lpm,allf)
    p=clf.predict_proba(sc.transform(F))[:,1]
    return torch.tensor(p.reshape(len(allf),T)).float()

rows={k:[] for k in ["noisy","model","global_p0.5","global_p0.7","gated_probe","gated_random","gated_oracle"]}
for i,fp in enumerate(test_f):
    a,_=sf.read(fp,dtype="float32"); a=torch.tensor(a[:int(3*SR)]); a=a/a.abs().max()*0.5
    rv=reverb(a,0.45,seed=i)                            # THE SHIFT: reverberant target+input
    L=rv.shape[0]
    for k in KINDS:
        for snr in [0,5]:
            nz=mk_noise(k,L,hash((i+700,k,snr))%9999); nz=nz/nz.std()*rv.std()/10**(snr/20)
            G,X,C,N,ibm,risk,st,lpm=analyse(rv,nz)
            U=full_risk(st,G,lpm)[:, :G.shape[1]]
            rows["noisy"].append(score((rv+nz)[:L],rv))
            rows["model"].append(score(synth(G,X,L),rv))
            for p in (0.5,0.7):
                rows[f"global_p{p}"].append(score(synth(G.pow(p).clamp(0.1,1),X,L),rv))
            for name,uu in [("gated_probe",U),
                            ("gated_random",U.flatten()[torch.randperm(U.numel())].reshape(U.shape)),
                            ("gated_oracle",(risk>0.5).float())]:
                rows[name].append(score(synth(G+uu*(1-G),X,L),rv))
print(f"{'system':>16} {'PESQ':>7} {'ESTOI':>7} {'SI-SDR':>8}   n={len(rows['model'])}  [REVERB shift]")
base=np.nanmean(np.array(rows['model']),axis=0)
for k,v in rows.items():
    mu=np.nanmean(np.array(v),axis=0)
    d="" if k=="model" else f"   dPESQ {mu[0]-base[0]:+.3f}"
    tag={"gated_random":"  <- control","gated_oracle":"  <- ceiling","global_p0.5":"  <- their knob"}.get(k,"")
    print(f"{k:>16} {mu[0]:7.3f} {mu[1]:7.3f} {mu[2]:8.2f}{d}{tag}")

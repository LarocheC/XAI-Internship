"""Is the gating probe's advantage from INTERNAL STATE, or just from resolution?

The risk probe used 64-d internal state + the bin's own gain, level and frequency. A fair
output-only baseline must get comparable capacity from the OUTPUT: the mask in a frequency
neighbourhood, plus level and frequency. If that matches the internal probe, the win is
about resolution, not internals -- the error the referee found in the earlier comparison.
"""
import sys, glob, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/user/XAI-Internship/DeepShap")
torch.manual_seed(0); np.random.seed(0)
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.model_utils import load_nsnet2_model
SR=16000; S="/tmp/claude-0/-home-user-XAI-Internship/d0cc0c4e-e629-5c87-9c02-44b0520c41b3/scratchpad"
m,_=load_nsnet2_model(); m.eval()

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

NB=8   # frequency neighbourhood half-width for the output baseline
def sample(clean,noise,fsub):
    noisy=clean+noise
    lp=m.log_power(m.preproc(noisy.unsqueeze(0)))
    h=lp.permute(0,2,1); h=m.fc1(h); g1,_=m.rnn1(h); g2,_=m.rnn2(g1)
    z=m.fc4(m.relu2(m.fc3(m.relu1(m.fc2(g2))))).permute(0,2,1)
    G=torch.sigmoid(z)[0].detach()
    C=m.preproc(clean.unsqueeze(0)).abs().squeeze(1)[0]
    N=m.preproc(noise.unsqueeze(0)).abs().squeeze(1)[0]
    T=min(G.shape[1],C.shape[1],N.shape[1]); G,C,N=G[:,:T],C[:,:T],N[:,:T]
    lpm=lp[0][:,:T].detach().numpy(); st=g2[0][:T].detach().numpy()
    y=((C>N).float()*(1-G)>0.5).numpy()
    Gp=np.pad(G.numpy(),((NB,NB),(0,0)),mode='edge')
    feats={}
    for f in fsub:
        nb=Gp[f:f+2*NB+1,:T].T                       # [T, 17] mask neighbourhood
        base=np.stack([lpm[f,:T],np.full(T,f/257.0)],1)
        feats.setdefault("A_internal64",[]).append(np.concatenate([st[:,:64],G[f,:T].numpy()[:,None],base],1))
        feats.setdefault("B_output_nbhd",[]).append(np.concatenate([nb,base],1))
        feats.setdefault("C_output+internal",[]).append(np.concatenate([st[:,:64],nb,base],1))
        feats.setdefault("D_gain_only",[]).append(np.concatenate([G[f,:T].numpy()[:,None],base],1))
    return {k:np.concatenate(v) for k,v in feats.items()}, np.concatenate([y[f,:T] for f in fsub])

files=sorted(glob.glob(f"{S}/LibriSpeech/test-clean/*/*/*.flac"))
FSUB=np.arange(4,256,10)
def build(fs,seed0):
    X={};Y=[]
    for i,fp in enumerate(fs):
        a,_=sf.read(fp,dtype="float32"); a=torch.tensor(a[:int(3*SR)]); a=a/a.abs().max()*0.5
        for k in ["white","pink","babble","hf"]:
            for snr in [0,5,10]:
                nz=mk_noise(k,a.shape[0],hash((seed0+i,k,snr))%9999); nz=nz/nz.std()*a.std()/10**(snr/20)
                f_,y=sample(a,nz,FSUB)
                for kk,v in f_.items(): X.setdefault(kk,[]).append(v)
                Y.append(y)
    return {k:np.concatenate(v) for k,v in X.items()}, np.concatenate(Y)
Xtr,ytr=build(files[:14],0); Xte,yte=build(files[40:58],500)
print(f"train {len(ytr):,} bins ({ytr.mean()*100:.1f}% pos), test {len(yte):,} "
      f"[held out by speaker]\n")
print(f"  {'features':>20} {'dim':>5} {'AUC':>7} {'AP':>7}   what it uses")
desc={"A_internal64":"64-d GRU state + own gain","B_output_nbhd":"mask in +-8 bins (OUTPUT ONLY)",
      "C_output+internal":"both","D_gain_only":"own gain only"}
res={}
for k in ["D_gain_only","B_output_nbhd","A_internal64","C_output+internal"]:
    sc=StandardScaler().fit(Xtr[k]); clf=LogisticRegression(max_iter=1200,C=0.1).fit(sc.transform(Xtr[k]),ytr)
    p=clf.predict_proba(sc.transform(Xte[k]))[:,1]
    res[k]=(roc_auc_score(yte,p),average_precision_score(yte,p))
    print(f"  {k:>20} {Xtr[k].shape[1]:5d} {res[k][0]:7.3f} {res[k][1]:7.3f}   {desc[k]}")
print(f"  {'chance':>20} {'-':>5} {0.5:7.3f} {yte.mean():7.3f}")
print(f"\n  internal vs fair output baseline: dAP {res['A_internal64'][1]-res['B_output_nbhd'][1]:+.3f}")
print(f"  adding internal on top of output: dAP {res['C_output+internal'][1]-res['B_output_nbhd'][1]:+.3f}")

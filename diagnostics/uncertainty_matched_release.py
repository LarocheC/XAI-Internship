"""Is the probe's win real, or just a different point on a release-amount tradeoff?

Softening a mask always trades noise-removal for speech-preservation. The global knob and
the probe release DIFFERENT AMOUNTS of gain, so comparing them at one setting each is not a
fair test: the probe may simply sit at a luckier point on the same curve.

Fix: sweep BOTH methods over release strength, measure the total release actually applied
(mean of g' - g), and compare the two curves AT MATCHED RELEASE. If the probe's curve
dominates the global curve everywhere, the signal is doing real work. If they lie on one
curve, the 'win' was an operating-point artefact.
"""
import sys, glob, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/user/XAI-Internship/DeepShap")
torch.manual_seed(0); np.random.seed(0)
exec(open("/tmp/claude-0/-home-user-XAI-Internship/d0cc0c4e-e629-5c87-9c02-44b0520c41b3/scratchpad/uncert2.py").read().split("rows={k:[]")[0])

STRENGTHS=[0.0,0.15,0.3,0.45,0.6,0.8,1.0]
POWERS=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3]
res={"global":[], "probe":[], "random":[], "oracle":[]}
for i,fp in enumerate(test_f[:16]):
    a,_=sf.read(fp,dtype="float32"); a=torch.tensor(a[:int(3*SR)]); a=a/a.abs().max()*0.5
    rv=reverb(a,0.45,seed=i); L=rv.shape[0]
    for k in ["white","babble"]:
        for snr in [0,5]:
            nz=mk_noise(k,L,hash((i+700,k,snr))%9999); nz=nz/nz.std()*rv.std()/10**(snr/20)
            G,X,C,N,ibm,risk,st,lpm=analyse(rv,nz)
            U=full_risk(st,G,lpm)[:,:G.shape[1]]
            Ur=U.flatten()[torch.randperm(U.numel())].reshape(U.shape)
            Uo=(risk>0.5).float()
            for p in POWERS:
                Gg=G.pow(p).clamp(0.1,1) if p<1.0 else G
                res["global"].append((float((Gg-G).mean()),)+score(synth(Gg,X,L),rv))
            for nm,uu in [("probe",U),("random",Ur),("oracle",Uo)]:
                for s in STRENGTHS:
                    Gg=(G+s*uu*(1-G)).clamp(0,1)
                    res[nm].append((float((Gg-G).mean()),)+score(synth(Gg,X,L),rv))
print("PESQ and ESTOI vs the AMOUNT of gain released (mean g'-g), reverb shift, n=64 clips/point\n")
for nm in ["global","probe","random","oracle"]:
    A=np.array(res[nm]); n=len(POWERS) if nm=="global" else len(STRENGTHS)
    A=A.reshape(-1,n,4)
    print(f"  {nm:>7}: " + "  ".join(
        f"rel={np.nanmean(A[:,j,0]):.3f} P={np.nanmean(A[:,j,1]):.3f} E={np.nanmean(A[:,j,2]):.3f}"
        for j in range(n)))
print("\nMatched-release comparison (interpolate each curve onto a common release axis):")
grid=np.linspace(0.02,0.30,7)
print(f"  {'release':>8} " + " ".join(f"{nm:>16}" for nm in ["global","probe","random","oracle"]))
for g_ in grid:
    line=f"  {g_:8.3f} "
    for nm in ["global","probe","random","oracle"]:
        A=np.array(res[nm]); n=len(POWERS) if nm=="global" else len(STRENGTHS)
        A=A.reshape(-1,n,4); rel=np.nanmean(A[:,:,0],axis=0); pq=np.nanmean(A[:,:,1],axis=0); es=np.nanmean(A[:,:,2],axis=0)
        o=np.argsort(rel)
        line+=f"  P={np.interp(g_,rel[o],pq[o]):.3f} E={np.interp(g_,rel[o],es[o]):.3f}"
    print(line)

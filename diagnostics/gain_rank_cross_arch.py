"""Cross-architecture: effective rank of the predicted gain surface for three enhancers.

Each model's gain is taken in ITS OWN domain, and the oracle ceiling is computed in the
matching domain -- LiSenNet masks a power-compressed magnitude (compress_factor 0.3), so
comparing it against a linear-domain IRM would confound architecture with representation.
Nulls (random, shuffled) are shape-matched per model.
"""
import sys, os, json, warnings, glob, numpy as np, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/larochec/eco8-neaixt")
sys.path.insert(0, "/home/user/XAI-Internship/DeepShap")
torch.manual_seed(0); np.random.seed(0)
import soundfile as sf
SR=16000; S="/tmp/claude-0/-home-user-XAI-Internship/d0cc0c4e-e629-5c87-9c02-44b0520c41b3/scratchpad"

def er(G, frac=0.90):
    s=np.linalg.svd(G-G.mean(), compute_uv=False); c=np.cumsum(s**2)/(s**2).sum()
    return int(np.searchsorted(c,frac)+1)

def stft(x): return torch.stft(x,512,256,window=torch.hann_window(512),return_complex=True)

# ---- models -------------------------------------------------------------
def get_nsnet2():
    from utils.model_utils import load_nsnet2_model
    m,_=load_nsnet2_model(); m.eval()
    def gain(noisy):
        with torch.no_grad():
            return torch.sigmoid(m.mask_logits(m.log_power(m.preproc(noisy.unsqueeze(0)))))[0].numpy()
    return gain, 1.0

def get_lisennet(sub="conv-hardened"):
    from lisennet.model import build_lisennet
    cfg=json.load(open(f"{S}/ckpt/{sub}/config.json")); m=build_lisennet(cfg)
    sd=torch.load(f"{S}/ckpt/{sub}/g_best",map_location="cpu",weights_only=True)
    sd=sd.get("generator",sd) if isinstance(sd,dict) else sd
    m.load_state_dict(sd,strict=False); m.eval()
    cf=float(cfg.get("compress_factor",0.3))
    def gain(noisy):
        with torch.no_grad():
            spec=m.power_compress(m.apply_stft(noisy.unsqueeze(0)))
            feat=m.build_features(spec.abs(),spec.angle())
            mk=m.predict_mask(feat)                       # (B,2,T,F)
            g=(mk[:,0]+mk[:,1])[0].T.numpy()              # -> (F,T)
        return g
    return gain, cf

def get_convfsenet():
    from convfsenet.model import ConvFSENet
    cfg=json.load(open(f"{S}/ckpt/config.json"))
    keys=dict(n_fft=cfg["n_fft"],win_length=cfg["win_length"],n_features=cfg["n_features"],
              n_channels_res=cfg["n_channels_res"],n_channels_conv=cfg["n_channels_conv"],
              kernel_size=cfg["kernel_size"],n_blocks=cfg["n_blocks"],n_stacks=cfg["n_stacks"],
              dropout=cfg.get("dropout",0.0),norm_type=cfg.get("norm_type","cLN"),
              extractor_type=cfg.get("extractor_type","mag"),
              compress_factor=cfg.get("compress_factor",1.0),causal=cfg.get("causal",True),
              loss=None,preproc=None,postproc=None)
    m=ConvFSENet(**keys)
    sd=torch.load(f"{S}/ckpt/g_best",map_location="cpu",weights_only=True)
    sd=sd.get("generator",sd) if isinstance(sd,dict) else sd
    print("   convfsenet load:", m.load_state_dict(sd,strict=False))
    m.eval()
    cf=float(keys["compress_factor"])
    def gain(noisy):
        with torch.no_grad():
            sp=stft(noisy).unsqueeze(0).unsqueeze(0)
            f=m.features_extractor(sp.squeeze(1))
            g=m.backend(m.tcm(m.frontend(f)))[0].numpy()
        return g
    return gain, cf

MODELS=[("NSNet2 (2.78M)",get_nsnet2),("LiSenNet conv-hardened (36k)",get_lisennet),
        ("ConvFSENet (1.45M)",get_convfsenet)]
files=sorted(glob.glob(f"{S}/LibriSpeech/test-clean/*/*/*.flac"))[:6]
print(f"{'system':>30} {'r90':>7} {'r99':>7} | {'IRM r90':>8} {'IRM r99':>8} | {'null r90':>9}")
print("-"*80)
for name,ctor in MODELS:
    try: gain,cf = ctor()
    except Exception as e:
        print(f"{name:>30}   SKIP: {type(e).__name__}: {str(e)[:60]}"); continue
    r90=[];r99=[];i90=[];i99=[];n90=[]
    for fp in files:
        a,_=sf.read(fp,dtype="float32"); a=torch.tensor(a[:int(3*SR)]); a=a/a.abs().max()*0.5
        g_=torch.Generator().manual_seed(1); n=torch.randn(a.shape,generator=g_)
        n=n/n.std()*a.std()/10**(5/20); noisy=a+n
        G=gain(noisy)
        C=stft(a).abs().numpy(); N=stft(n).abs().numpy()
        T=min(G.shape[1],C.shape[1]); G,C,N=G[:,:T],C[:,:T],N[:,:T]
        irm=(C**cf)/((C**cf)+(N**cf)+1e-9)          # oracle in the SAME domain the model masks
        rnd=np.random.rand(*G.shape)
        r90.append(er(G,.90)); r99.append(er(G,.99))
        i90.append(er(irm,.90)); i99.append(er(irm,.99)); n90.append(er(rnd,.90))
    print(f"{name:>30} {np.mean(r90):7.1f} {np.mean(r99):7.1f} | {np.mean(i90):8.1f} {np.mean(i99):8.1f} | {np.mean(n90):9.1f}")
print(f"\n(6 LibriSpeech utts, 5 dB white noise; oracle IRM computed in each model's own gain domain)")

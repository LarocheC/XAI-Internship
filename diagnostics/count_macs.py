"""Count LiSenNet's per-frame MACs by hooks, and compare to its parameter count.

Checks the claim 'LiSenNet is 19x smaller, so explanation on it is cheaper' -- which is
about MACs, not parameters, and for a conv net those diverge sharply.
"""
import sys, json, warnings, torch, torch.nn as nn
warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/user/larochec/eco8-neaixt")
from lisennet.model import build_lisennet
S="/tmp/claude-0/-home-user-XAI-Internship/d0cc0c4e-e629-5c87-9c02-44b0520c41b3/scratchpad"
cfg=json.load(open(f"{S}/ckpt/conv-hardened/config.json")); m=build_lisennet(cfg)
sd=torch.load(f"{S}/ckpt/conv-hardened/g_best",map_location="cpu",weights_only=True)
m.load_state_dict(sd.get("generator",sd) if isinstance(sd,dict) else sd,strict=False); m.eval()
P=sum(p.numel() for p in m.parameters())

tot={"n":0}
def hook(mod,inp,out):
    if isinstance(mod,nn.Conv2d):
        o=out.shape; tot["n"]+= (mod.in_channels//mod.groups)*mod.kernel_size[0]*mod.kernel_size[1]*o[1]*o[2]*o[3]
    elif isinstance(mod,nn.Conv1d):
        o=out.shape; tot["n"]+= (mod.in_channels//mod.groups)*mod.kernel_size[0]*o[1]*o[2]
    elif isinstance(mod,nn.Linear):
        tot["n"]+= mod.in_features*mod.out_features*int(torch.tensor(out.shape[:-1]).prod())
    elif isinstance(mod,nn.GRU):
        i=inp[0].shape; L=i[1] if mod.batch_first else i[0]
        tot["n"]+= 3*(mod.input_size*mod.hidden_size+mod.hidden_size*mod.hidden_size)*L*(2 if mod.bidirectional else 1)
hs=[mod.register_forward_hook(hook) for mod in m.modules()]
T_FRAMES=100
x=torch.randn(1,int(T_FRAMES*256))*0.05
with torch.no_grad(): m(x)
for h in hs: h.remove()
per_frame=tot["n"]/T_FRAMES
print(f"LiSenNet conv-hardened nc24")
print(f"  parameters      {P:12,}")
print(f"  MACs per frame  {per_frame:12,.0f}   (measured by hooks over {T_FRAMES} frames)")
print(f"  MAC/param ratio {per_frame/P:12.1f}x   <- convolutional reuse")
print()
print(f"NSNet2 blockdiag_full (FC+GRU: each weight used once per frame, so MACs ~ params)")
print(f"  parameters      {701657:12,}")
print(f"  MACs per frame  {701657:12,}   (analytic)")
print()
print(f"=> LiSenNet has {701657/P:.1f}x FEWER PARAMETERS but "
      f"{per_frame/701657:.2f}x the per-frame ARITHMETIC.")

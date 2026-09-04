"""Does r90 on a gain surface actually recover known effective rank?

The discipline RETRACTION.md demands: build surfaces whose rank is set by construction,
push them through the identical estimator, and check it reads them back.
"""
import numpy as np
np.random.seed(0)
F, T = 257, 187

def er(G, frac=0.90):
    s = np.linalg.svd(G - G.mean(), compute_uv=False)
    c = np.cumsum(s**2)/(s**2).sum()
    return int(np.searchsorted(c, frac)+1)

print(f"{'constructed rank k':>20} {'r90 read back':>14} {'r99':>8}")
for k in [1,2,4,8,16,32,64,128,257]:
    U=np.random.randn(F,k); V=np.random.randn(k,T)
    G=U@V
    G=(G-G.min())/(G.max()-G.min())          # squash into [0,1] like a gain, a mild nonlinearity
    print(f"{k:>20} {er(G):>14} {er(G,0.99):>8}")
print()
print("same, but passed through a sigmoid (as a real gain surface is):")
print(f"{'constructed rank k':>20} {'r90 read back':>14} {'r99':>8}")
for k in [1,2,4,8,16,32,64,128,257]:
    U=np.random.randn(F,k); V=np.random.randn(k,T)
    G=1/(1+np.exp(-(U@V)/np.sqrt(k)))
    print(f"{k:>20} {er(G):>14} {er(G,0.99):>8}")

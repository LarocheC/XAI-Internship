"""Does my estimator R[f_in,f_out] = sum_t |A_t| distinguish rank at all?

Feed it synthetic Jacobians of KNOWN rank and see what it reports.
"""
import numpy as np
np.random.seed(0)
F, T = 257, 60

def stats(R, out_idx, band=8):
    sv = np.linalg.svd(R, compute_uv=False); s2 = sv**2
    top1 = float(s2[0]/s2.sum()); pr = float((s2.sum()**2)/(s2**2).sum())
    fi = np.arange(R.shape[0])[:,None]; fo = np.asarray(out_idx)[None,:]
    near = np.abs(fi-fo) <= band
    enr = float((R[near].sum()/R.sum())/(near.sum()/R.size))
    return top1, pr, enr

out_idx = np.linspace(1, F-1, 64).astype(int)
n_out = len(out_idx)

def make(kind):
    """Return A of shape [F_in, n_out, T] with known structure."""
    if kind.startswith("rank"):
        k = int(kind[4:])
        U = np.random.randn(F, k, T); V = np.random.randn(k, n_out, T)
        return np.einsum('ikt,kot->iot', U, V)
    if kind == "fullrank":
        return np.random.randn(F, n_out, T)
    if kind == "diagonal":
        A = np.zeros((F, n_out, T))
        for j, f in enumerate(out_idx): A[f, j, :] = np.random.randn(T)
        return A

print(f"{'true structure':>16} | {'ABS-then-sum (mine)':>34} | {'signed sum':>22}")
print(f"{'':>16} | {'top1':>7} {'PR':>8} {'enrich':>8} {'':>7} | {'top1':>7} {'PR':>8}")
print("-"*80)
for kind in ["fullrank","rank64","rank8","rank1","diagonal"]:
    A = make(kind)
    R_abs = np.abs(A).sum(axis=2)                 # exactly what my code does
    R_sgn = np.abs(A.sum(axis=2))                 # sum first, then abs
    a = stats(R_abs, out_idx); b = stats(R_sgn, out_idx)
    print(f"{kind:>16} | {a[0]:7.3f} {a[1]:8.2f} {a[2]:8.2f} {'':>7} | {b[0]:7.3f} {b[1]:8.2f}")
print()
print("geometric ceiling of the enrichment metric (a perfectly diagonal matrix):")
fi=np.arange(F)[:,None]; fo=out_idx[None,:]; near=np.abs(fi-fo)<=8
print(f"   1/(area fraction) = {1.0/(near.sum()/ (F*n_out)):.2f}   <- compare to the 'oracle Wiener control' value")

"""SVD operator — Triton-native paths for torch.svd.

Paths:
- K<=8:  register-based fused Jacobi SVD kernel
- K>8 :  Gram via Triton tl.dot + Jacobi eigendecomposition via Triton kernel
"""

import math, logging
from collections import namedtuple
import torch, triton, triton.language as tl
from flag_gems.utils import libentry, triton_lang_extension as tle

logger = logging.getLogger(__name__)
_FALLBACK_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd)

def _fallback_svd(input, full_matrices=True):
    return torch.ops.aten.linalg_svd.default.redispatch(_FALLBACK_KEYSET, input, full_matrices)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_fp32_cuda_mat(x): return x.is_cuda and x.dtype == torch.float32 and x.ndim >= 2
def _svd_dims(x):
    if x.ndim < 2: return 0,0,0
    m,n = x.shape[-2],x.shape[-1]
    b = 1
    for d in x.shape[:-2]: b *= d
    return b,m,n
def _can_rank1(x):
    _,m,n = _svd_dims(x); return _is_fp32_cuda_mat(x) and min(m,n)==1
def _can_small_jacobi(x):
    _,m,n = _svd_dims(x); return _is_fp32_cuda_mat(x) and min(m,n)<=8 and max(m,n)<=1024

# ---------------------------------------------------------------------------
# Brent-Luk pairs for Jacobi eigendecomposition
# ---------------------------------------------------------------------------
def _brent_luk_pairs(K):
    if K <= 1: return []
    steps = []
    n_eff = K if K%2==0 else K+1
    for s in range(n_eff-1):
        i_l, j_l = [], []
        for k in range(n_eff//2):
            i = (s+k)%(n_eff-1); j = n_eff-1 if k==0 else (s+n_eff-1-k)%(n_eff-1)
            if i<K and j<K: i_l.append(i); j_l.append(j)
        if i_l: steps.append((i_l, j_l))
    return steps

# ---------------------------------------------------------------------------┴
# rank-1 kernel
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _rank1_kernel(x,u,s,vh,b,M:tl.constexpr,N:tl.constexpr,T:tl.constexpr,BM:tl.constexpr):
    pid=tle.program_id(0); rows=tl.arange(0,BM); sz=tl.maximum(M,N)
    vals=tl.load(x+pid*M*N+(rows if T else rows*N),mask=rows<sz,other=0.0).to(tl.float32)
    sq=tl.sum(vals*vals,axis=0); nrm=tl.sqrt(tl.maximum(sq,0.0)); inv=1.0/tl.sqrt(tl.maximum(sq,1e-30))
    uvals=vals*inv; tl.store(s+pid,nrm)
    if T: tl.store(u+pid*M+rows,uvals,mask=rows<sz); tl.store(vh+pid,1.0)
    else:  tl.store(u+pid,1.0); tl.store(vh+pid*N+rows,uvals,mask=rows<sz)

def _rank1(x,full=True):
    b,m,n=_svd_dims(x); tall=m>=n; sz=max(m,n); dev,dt=x.device,x.dtype
    u=torch.empty(b,m,1,device=dev,dtype=dt); vh=torch.empty(b,1,n,device=dev,dtype=dt)
    s=torch.empty(b,1,device=dev,dtype=torch.float32)
    _rank1_kernel[(b,)](x,u,s,vh,b,m,n,tall,BM=min(1024,triton.next_power_of_2(sz)))
    return u,s,vh.mT

# ---------------------------------------------------------------------------
# small Jacobi SVD (K <= 8, register-based)
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _small_jacobi_kernel(A,aw,vw,U,S,Vh,M:tl.constexpr,K:tl.constexpr,T:tl.constexpr,
                          SW:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr):
    pid=tle.program_id(0); rows=tl.arange(0,BM); cols=tl.arange(0,BK)
    rm=rows<M; cm=cols<K; eps=1e-20
    ab=A+pid*M*K; awb=aw+pid*K*M; vwb=vw+pid*K*K
    for j in tl.static_range(0,K):
        v=tl.load(ab+(rows*K+j if T else j*M+rows),mask=rm,other=0.0).to(tl.float32)
        tl.store(awb+j*M+rows,v,mask=rm); tl.store(vwb+j*K+cols,tl.where(cols==j,1.0,0.0),mask=cm)
    for _ in tl.static_range(0,SW):
        for p in tl.static_range(0,K):
            for q in tl.static_range(p+1,K):
                ap=tl.load(awb+p*M+rows,mask=rm,other=0.0); aq=tl.load(awb+q*M+rows,mask=rm,other=0.0)
                al=tl.sum(ap*ap); be=tl.sum(aq*aq); ga=tl.sum(ap*aq)
                ac=tl.abs(ga)>1e-7*tl.sqrt(al*be+eps)
                tau=(be-al)/(2.0*ga+1e-30); sg=tl.where(tau>=0,1.0,-1.0)
                t_=sg/(tl.abs(tau)+tl.sqrt(1+tau*tau)); c=tl.where(ac,tl.rsqrt(1+t_*t_),1.0); s=tl.where(ac,t_*c,0.0)
                tl.store(awb+p*M+rows,c*ap-s*aq,mask=rm); tl.store(awb+q*M+rows,s*ap+c*aq,mask=rm)
                vp=tl.load(vwb+p*K+cols,mask=cm,other=0.0); vq=tl.load(vwb+q*K+cols,mask=cm,other=0.0)
                tl.store(vwb+p*K+cols,c*vp-s*vq,mask=cm); tl.store(vwb+q*K+cols,s*vp+c*vq,mask=cm)
    for col in tl.static_range(0,K):
        v=tl.load(awb+col*M+rows,mask=rm,other=0.0); sq=tl.sum(v*v,axis=0)
        nm=tl.sqrt(tl.maximum(sq,0.0)); tl.store(S+pid*K+col,nm)
        inv=1.0/tl.sqrt(tl.maximum(sq,1e-30)); tl.store(U+pid*M*K+rows*K+col,v*inv,mask=rm)
    for col in tl.static_range(0,K):
        v=tl.load(vwb+col*K+cols,mask=cm,other=0.0); tl.store(Vh+pid*K*K+cols*K+col,v,mask=cm)

def _small_jacobi(x,full=True):
    b,m,n=_svd_dims(x); dev,dt=x.device,x.dtype; tall=m>=n; M=max(m,n); K=min(m,n)
    A=x if tall else x.transpose(-2,-1)
    if A.ndim==2: A=A.unsqueeze(0).contiguous()
    else: A=A.reshape(b,M,K).contiguous()
    aw=torch.empty(b,K,M,device=dev,dtype=torch.float32)
    vw=torch.empty(b,K,K,device=dev,dtype=torch.float32)
    U=torch.empty(b,M,K,device=dev,dtype=dt); S=torch.empty(b,K,device=dev,dtype=torch.float32)
    Vh=torch.empty(b,K,K,device=dev,dtype=dt)
    BM=min(1024,triton.next_power_of_2(M)); SW=min(K,12)
    _small_jacobi_kernel[(b,)](A,aw,vw,U,S,Vh,M=M,K=K,T=True,SW=SW,BM=BM,BK=triton.next_power_of_2(K),
                                num_warps=4 if M<=64 else 8)
    S,idx=S.sort(dim=-1,descending=True); U=U.gather(-1,idx.unsqueeze(-2).expand_as(U))
    Sinv=1.0/S.clamp_min(1e-30); Vh=Sinv.unsqueeze(-1)*U.float().transpose(-2,-1)@A.float()
    V=Vh.mT.to(dt)
    if not tall: U,V=V,U
    return U,S,V

# ---------------------------------------------------------------------------
# Gram kernel (Triton tl.dot)
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _gram_kernel(x,g,b,M:tl.constexpr,N:tl.constexpr,BN:tl.constexpr,BM:tl.constexpr):
    pb=tle.program_id(0); pi=tle.program_id(1); pj=tle.program_id(2)
    oi=pi*BN+tl.arange(0,BN); oj=pj*BN+tl.arange(0,BN); om=tl.arange(0,BM)
    mi=oi<N; mj=oj<N; acc=tl.zeros((BN,BN),dtype=tl.float32)
    for m0 in range(0,M,BM):
        m=m0+om; mm=m<M
        ai=tl.load(x+pb*M*N+m[:,None]*N+oi[None,:],mask=mm[:,None]&mi[None,:],other=0.0).to(tl.float32)
        aj=tl.load(x+pb*M*N+m[:,None]*N+oj[None,:],mask=mm[:,None]&mj[None,:],other=0.0).to(tl.float32)
        acc+=tl.dot(tl.trans(ai),aj,input_precision="ieee")
    tl.store(g+pb*N*N+oi[:,None]*N+oj[None,:],acc,mask=mi[:,None]&mj[None,:])

# ---------------------------------------------------------------------------
# Jacobi eigendecomposition of G (b, K, K) via Triton kernels
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _jacobi_eig_row_kernel(
    G_ptr, K: tl.constexpr,
    i_idx_ptr, j_idx_ptr, c_ptr, s_ptr, num_pairs: tl.constexpr, BLK: tl.constexpr,
):
    pid = tle.program_id(0)
    pair_id = pid % num_pairs; batch_id = pid // num_pairs
    ii = tl.load(i_idx_ptr + pair_id).to(tl.int32); jj = tl.load(j_idx_ptr + pair_id).to(tl.int32)

    # Compute rotation from G entries (inside kernel, no Python GPU indexing)
    g_off = batch_id * K * K
    g_pp = tl.load(G_ptr + g_off + ii * K + ii).to(tl.float32)
    g_qq = tl.load(G_ptr + g_off + jj * K + jj).to(tl.float32)
    g_pq = tl.load(G_ptr + g_off + ii * K + jj).to(tl.float32)
    tau = (g_pp - g_qq) / (2.0 * g_pq + 1e-30)
    sg = tl.where(tau >= 0.0, 1.0, -1.0)
    t_val = sg / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    c_val = tl.rsqrt(1.0 + t_val * t_val)
    s_val = t_val * c_val

    # Output c, s for column+V kernel
    tl.store(c_ptr + pid, c_val)
    tl.store(s_ptr + pid, s_val)

    # Row update: G = R^T @ G
    for k in range(0, K, BLK):
        off = k + tl.arange(0, BLK); mask = off < K
        gi = tl.load(G_ptr + g_off + ii * K + off, mask=mask, other=0.0).to(tl.float32)
        gj = tl.load(G_ptr + g_off + jj * K + off, mask=mask, other=0.0).to(tl.float32)
        tl.store(G_ptr + g_off + ii * K + off, c_val * gi + s_val * gj, mask=mask)
        tl.store(G_ptr + g_off + jj * K + off, -s_val * gi + c_val * gj, mask=mask)

@libentry()
@triton.jit
def _jacobi_eig_col_kernel(
    G_ptr, V_ptr, K: tl.constexpr,
    i_idx_ptr, j_idx_ptr, c_ptr, s_ptr, num_pairs: tl.constexpr, BLK: tl.constexpr,
):
    pid = tle.program_id(0)
    pair_id = pid % num_pairs; batch_id = pid // num_pairs
    ii = tl.load(i_idx_ptr + pair_id).to(tl.int32); jj = tl.load(j_idx_ptr + pair_id).to(tl.int32)
    c_val = tl.load(c_ptr + pid).to(tl.float32)
    s_val = tl.load(s_ptr + pid).to(tl.float32)
    g_off = batch_id * K * K; v_off = batch_id * K * K
    for k in range(0, K, BLK):
        off = k + tl.arange(0, BLK); mask = off < K
        gi = tl.load(G_ptr + g_off + off * K + ii, mask=mask, other=0.0).to(tl.float32)
        gj = tl.load(G_ptr + g_off + off * K + jj, mask=mask, other=0.0).to(tl.float32)
        tl.store(G_ptr + g_off + off * K + ii, c_val * gi + s_val * gj, mask=mask)
        tl.store(G_ptr + g_off + off * K + jj, -s_val * gi + c_val * gj, mask=mask)
    for k in range(0, K, BLK):
        off = k + tl.arange(0, BLK); mask = off < K
        vi = tl.load(V_ptr + v_off + off * K + ii, mask=mask, other=0.0).to(tl.float32)
        vj = tl.load(V_ptr + v_off + off * K + jj, mask=mask, other=0.0).to(tl.float32)
        tl.store(V_ptr + v_off + off * K + ii, c_val * vi + s_val * vj, mask=mask)
        tl.store(V_ptr + v_off + off * K + jj, -s_val * vi + c_val * vj, mask=mask)


def _jacobi_eigh_gpu(G, max_sweeps=5):
    batch, K, _ = G.shape
    device, dtype = G.device, G.dtype
    G_work = G.float().clone()
    V = torch.eye(K, device=device, dtype=torch.float32).unsqueeze(0).expand(batch, K, K).clone()

    steps = _brent_luk_pairs(K)
    if not steps:
        return G.diagonal(dim1=-2, dim2=-1).clamp_min(0.0), V.to(dtype)

    # Pre-build index tensors for all steps
    step_tensors = []
    for i_l, j_l in steps:
        step_tensors.append((
            torch.tensor(i_l, device=device, dtype=torch.int32),
            torch.tensor(j_l, device=device, dtype=torch.int32),
            len(i_l)
        ))

    BLK = 64
    # Single c/s buffer reused across all steps (max size = K/2 * batch)
    max_pairs = max(len(il) for il, _ in steps)
    cs_buf_c = torch.empty(batch * max_pairs, device=device, dtype=torch.float32)
    cs_buf_s = torch.empty(batch * max_pairs, device=device, dtype=torch.float32)
    for _ in range(max_sweeps):
        for i_t, j_t, npairs in step_tensors:
            _jacobi_eig_row_kernel[(npairs * batch,)](G_work, K, i_t, j_t, cs_buf_c, cs_buf_s,
                                                       num_pairs=npairs, BLK=BLK)
            _jacobi_eig_col_kernel[(npairs * batch,)](G_work, V, K, i_t, j_t, cs_buf_c, cs_buf_s,
                                                       num_pairs=npairs, BLK=BLK)

    S_sq = G_work.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
    return S_sq, V.to(dtype)


def _svd_gram_jacobi(x, full=True):
    b,m,n=_svd_dims(x); dev,dt=x.device,x.dtype; tall=m>=n; M=max(m,n); K=min(m,n)
    if tall: X=x.reshape(b,m,n)
    else:    X=x.reshape(b,m,n).transpose(-2,-1)

    gram=torch.zeros(b,K,K,device=dev,dtype=torch.float32)
    grid=(b,triton.cdiv(K,32),triton.cdiv(K,32))
    _gram_kernel[grid](X,gram,b,M=M,N=K,BN=32,BM=64,num_warps=4,num_stages=2)

    S_sq,V_cols=_jacobi_eigh_gpu(gram)
    S_sq,idx=S_sq.sort(dim=-1,descending=True)
    V_cols=V_cols.gather(-1,idx.unsqueeze(-2).expand_as(V_cols))
    S=S_sq.clamp_min(0.0).sqrt()
    U=(X.float()@V_cols.float())/S.unsqueeze(-2).clamp_min(1e-30)
    U=U.to(dt); V=V_cols.to(dt)
    U,S,V=U,S,V
    if not tall: U,V=V,U
    return U,S,V

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def svd(input, some=True, compute_uv=True):
    fm = not some
    if not compute_uv:
        S=torch.linalg.svdvals(input); m,n=input.shape[-2],input.shape[-1]
        U=torch.zeros(*input.shape[:-2],m,m,device=input.device,dtype=input.dtype)
        V=torch.zeros(*input.shape[:-2],n,n,device=input.device,dtype=input.dtype)
        return U,S,V
    if not _is_fp32_cuda_mat(input) or not some or min(input.shape[-2],input.shape[-1])==0:
        U,S,Vh=_fallback_svd(input,fm); return U,S,Vh.mH
    was2d=input.ndim==2; obp=input.shape[:-2]
    if was2d: aw=input.unsqueeze(0)
    else:     aw=input.reshape(-1,*input.shape[-2:])
    aw=aw.contiguous()
    if _can_rank1(aw):        U,S,V=_rank1(aw)
    elif _can_small_jacobi(aw): U,S,V=_small_jacobi(aw)
    else:                      U,S,V=_svd_gram_jacobi(aw)
    if was2d: U,S,V=U.squeeze(0),S.squeeze(0),V.squeeze(0)
    elif len(obp)>0:
        U=U.reshape(*obp,*U.shape[-2:]); S=S.reshape(*obp,S.shape[-1]); V=V.reshape(*obp,*V.shape[-2:])
    return U,S,V

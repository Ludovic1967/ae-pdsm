# -*- coding: utf-8 -*-
"""
Y-binned 可视化（期刊规范 | 交换坐标轴）+ 保存每分组首个 (X,Y)
- (Train+Test) 真实 (X,Y)：中位趋势 + IQR 带 + LOWESS（X=分箱, Y=sX）
- (Test) 预测 (X,Ŷ)：竖直箱线图叠加
- 每工况一箱、升/降序、中心值标签、对比增强（probit/yeo/Δ）、symlog
- 期刊级输出：Times、cm→inch、1200 dpi、TIFF/LZW
- 新增：保存每个分组中的**第一个样本对 (X,Y)** 的 .npy 和图片
- 新增：从 X 抽取 3 个光谱特征（HFER/SC/SBW），分别与 sY、与 sX 画散点 + Spearman ρ
"""
import os, argparse, math, warnings
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer
from matplotlib import cm
import imageio

# ==== 0. 期刊输出规范 ====
journal = {
    "fig_size": (25/2.54, 14/2.54),   # cm → inch
    "font_name": "Arial",
    "font_size": 15,
    "line_width": 1.5,
    "axis_line_width": 1.0,
    "dpi": 1200
}
SAVE_EXT = "tiff"  # 可改 "png" / "pdf" / "svg" / "tiff"

def apply_journal_style(j):
    mpl.rcParams.update({
        "figure.figsize": j["fig_size"],
        "savefig.dpi": j["dpi"],
        "font.family": j["font_name"],
        "font.size": j["font_size"],
        "axes.labelsize": j["font_size"],
        "axes.titlesize": j["font_size"],
        "xtick.labelsize": j["font_size"],
        "ytick.labelsize": j["font_size"],
        "legend.fontsize": j["font_size"],
        "lines.linewidth": j["line_width"],
        "axes.linewidth": j["axis_line_width"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": j["axis_line_width"],
        "ytick.major.width": j["axis_line_width"],
        "xtick.minor.width": j["axis_line_width"],
        "ytick.minor.width": j["axis_line_width"],
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # "mathtext.fontset": "stix",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",  # 正体 roman
        "mathtext.it": "Arial:italic",  # 斜体
        "mathtext.bf": "Arial:bold",  # 粗体
        "mathtext.sf": "Arial",  # 无衬线
    })

def savefig_journal(fig, path, j, ext="tiff"):
    ext = ext.lower()
    kwargs = dict(dpi=j["dpi"], bbox_inches="tight")
    if ext in ("tif", "tiff"):
        kwargs.update(format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path, **kwargs)

apply_journal_style(journal)

# ======== 可选 LOWESS（自动回退到样条） ========
def lowess_smooth(x, y, frac=0.35):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(ys, xs, frac=frac, return_sorted=True)
        return sm[:,0], sm[:,1]
    except Exception:
        try:
            from scipy.interpolate import UnivariateSpline
            s = len(xs)*np.var(ys)*0.1
            spl = UnivariateSpline(xs, ys, s=s)
            xs2 = np.linspace(xs.min(), xs.max(), 200)
            return xs2, spl(xs2)
        except Exception:
            warnings.warn("LOWESS/样条均不可用，将跳过平滑线。")
            return np.array([]), np.array([])

# ====================== 通用工具 ======================
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def unpack_batch(batch):
    """支持 (X, Y) 或 dict 含 'X','Y'。"""
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        else:
            raise ValueError("Batch tuple/list length < 2.")
    elif isinstance(batch, dict):
        for kx in ['X','x','inputs','signals']:
            if kx in batch:
                X = batch[kx]; break
        else:
            raise KeyError("Cannot find X in batch dict.")
        for ky in ['Y','y','label','labels','target','targets']:
            if ky in batch:
                Y = batch[ky]; break
        else:
            raise KeyError("Cannot find Y in batch dict.")
        return X, Y
    else:
        raise TypeError("Unsupported batch type.")

def normalize_y_shape(arr):
    """
    将 Y or Ŷ 变为 (H, W) 的 numpy 数组（单样本时）。
    也可处理 (N,1,H,W)/(N,H,W)/(N,H,W,1) 的批。
    """
    A = to_numpy(arr)
    if A.ndim == 4:
        if A.shape[1] == 1:
            A = A[:, 0, :, :]
        elif A.shape[-1] == 1:
            A = A[..., 0]
    return A

# ====================== sY 指标（面积/紧致度/RSI） ======================
def _binary_mask(y2d):
    y = to_numpy(y2d).astype(float)
    u = np.unique(y)
    if u.size <= 3 and set(np.round(u).tolist()).issubset({0,1}):
        return (y > 0.5)
    try:
        from skimage.filters import threshold_otsu
        thr = threshold_otsu(y)
    except Exception:
        thr = np.median(y)
    return (y >= thr)

def _perimeter_px(mask):
    m = mask.astype(np.uint8)
    ph = np.sum(m[:,1:] != m[:,:-1])
    pv = np.sum(m[1:,:] != m[:-1,:])
    return float(ph + pv)

def _rsi(mask):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return np.nan
    c = idx.mean(axis=0)
    r = np.linalg.norm(idx - c[None,:], axis=1)
    return float(r.std())

def compute_sy_metrics(y2d):
    mask = _binary_mask(y2d)
    A = float(mask.sum())
    if A <= 0:
        return dict(area=0.0, compactness=np.nan, rsi=np.nan)
    P = _perimeter_px(mask)
    compact = 4*math.pi*A / (P**2 + 1e-9)
    rsi_val = _rsi(mask)
    return dict(area=A, compactness=compact, rsi=rsi_val)

def pick_sy(mets, primary_sy):
    if primary_sy == 'area': return mets['area']
    if primary_sy == 'compactness': return mets['compactness']
    if primary_sy == 'rsi': return mets['rsi']
    raise ValueError("primary_sy must be in {'area','compactness','rsi'}")

# ====================== 标准化（robust） ======================
class RobustStandardizer:
    def __init__(self):
        self.med = None
        self.scale = None
        self.fallback = False
        self.mean = None
        self.std = None
    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.med = np.median(x)
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr < 1e-8:
            self.fallback = True
            self.mean = float(np.mean(x))
            self.std = float(np.std(x) + 1e-9)
        else:
            self.scale = float(iqr)
        return self
    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if self.fallback:
            return (x - self.mean) / self.std
        else:
            return (x - self.med) / self.scale

# ====================== 抓取潜变量 z → sX ======================
def find_module(model, name='encoder'):
    for n, m in model.named_modules():
        if n.endswith(name) or n == name:
            return m
    return None

def last_conv_module(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

class _FeatureCatcher:
    def __init__(self):
        self.out = None
    def __call__(self, module, inp, out):
        self.out = out

def extract_latent_batch(model, x, module_name='encoder'):
    model.eval()
    hook_mod = find_module(model, module_name)
    catcher = _FeatureCatcher()

    if hook_mod is not None:
        h = hook_mod.register_forward_hook(catcher)
        with torch.no_grad():
            _ = model(x)
        h.remove()
        z = catcher.out
        if isinstance(z, (list, tuple)):
            z = z[0]
        if z.dim() == 4: z = z.mean(dim=[2,3])
        elif z.dim() == 3: z = z.mean(dim=2)
        return z
    else:
        lastc = last_conv_module(model)
        if lastc is None:
            raise RuntimeError("未找到 encoder 或 Conv2d，无法提取 z。请指定 module_name。")
        h = lastc.register_forward_hook(catcher)
        with torch.no_grad():
            _ = model(x)
        h.remove()
        z = catcher.out
        if z.dim() == 4: z = z.mean(dim=[2,3])
        elif z.dim() == 3: z = z.mean(dim=2)
        return z

from sklearn.decomposition import PCA
import numpy as np
import warnings

def sx_from_z_train(z_mat, mode='pc1', verbose=True, label=''):
    """
    z_mat: (N, D)
    mode: 'pc1' or 'l2'
    verbose: True 时打印 PC1 的 EVR
    label: 打印时的标识（如 'fold1-train' 或 'fold1-all'）

    return: sX (N,), pca
    其中 pca.evr_ = PC1 的 EVR (float)
    """
    Z = np.asarray(z_mat, dtype=float)

    if mode == 'l2':
        sX = np.linalg.norm(Z, axis=1)
        if verbose:
            msg = f"[PCA]{(' '+label) if label else ''} mode='l2'：无需PCA，EVR不可用"
            print(msg)
        # 为了接口一致，仍返回 None；也可挂个属性以防外部访问
        class _Dummy: pass
        pca = _Dummy()
        pca.evr_ = None
        return sX, pca

    # PCA-1
    pca = PCA(n_components=1, random_state=0)
    sX = pca.fit_transform(Z)[:, 0]

    # 解释方差比（相对于总方差）
    try:
        evr = float(pca.explained_variance_ratio_[0])
    except Exception:
        evr = np.nan
        warnings.warn("无法读取 explained_variance_ratio_，EVR=NaN")

    # 便于外部读取
    pca.evr_ = evr

    if verbose:
        n, d = Z.shape
        msg = (f"[PCA]{(' '+label) if label else ''} "
               f"PC1 EVR={evr:.4f} ({evr*100:.2f}%)  "
               f"n_samples={n}, n_features={d}")
        print(msg)

    return sX, pca


def sx_from_z_apply(z_mat, pca_model=None, mode='pc1'):
    Z = np.asarray(z_mat, dtype=float)
    if mode == 'l2' or pca_model is None:
        return np.linalg.norm(Z, axis=1) if mode=='l2' else Z[:,0]
    return pca_model.transform(Z)[:,0]

def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    uniq, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt + 1
    avg = (start + csum) / 2.0
    return avg[inv]

def spearman_rho(a, b):
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(a, b, nan_policy='omit')
        return float(r)
    except Exception:
        a = np.asarray(a); b = np.asarray(b)
        def rank_avg(x):
            order = np.argsort(x, kind='mergesort')
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(x), dtype=float)
            vals, idx_start = np.unique(x[order], return_index=True)
            for s, v in zip(idx_start, vals):
                e = s + np.sum(x[order]==v)
                avg = (s + e - 1) / 2.0
                ranks[order[s:e]] = avg
            return ranks
        ra, rb = rank_avg(a), rank_avg(b)
        if np.std(ra)==0 or np.std(rb)==0: return np.nan
        return float(np.corrcoef(ra, rb)[0,1])

def scatter_with_rho(x_tr, y_tr, x_te, y_te,
                     xlabel, ylabel, out_path_with_ext, title='',
                     annotate=True,
                     fig_size=None,         # 新增：英寸 (W, H)，如 (6, 4)
                     fig_size_cm=None,      # 新增：厘米 (W, H)，如 (12, 9)
                     dpi_override=None,     # 新增：仅对本图保存时覆盖 DPI（可选）
                     use_journal_save=True  # 新增：是否用 savefig_journal 保存
                     ):
    import numpy as np, matplotlib.pyplot as plt

    # —— 计算关联系数（原逻辑不变）——
    x_tr = np.asarray(x_tr); y_tr = np.asarray(y_tr)
    x_te = np.asarray(x_te); y_te = np.asarray(y_te)
    r_tr = spearman_rho(x_tr, y_tr)
    r_te = spearman_rho(x_te, y_te)
    r_all = spearman_rho(np.concatenate([x_tr, x_te]),
                         np.concatenate([y_tr, y_te]))

    # —— 尺寸处理：fig_size_cm 优先生效，其次 fig_size，最后用全局默认 ——
    if fig_size_cm is not None:
        fig_size = (fig_size_cm[0]/2.54, fig_size_cm[1]/2.54)

    # 只对这个图生效的局部尺寸（不会改全局 rcParams）
    if fig_size is not None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig, ax = plt.subplots()  # 用 journal 的全局默认

    # —— 作图（原逻辑不变）——
    # ax.scatter(x_tr, y_tr, s=18, alpha=0.5, label=f"Train (ρ={r_tr:.2f})")
    # ax.scatter(x_te, y_te, s=24, alpha=0.9, marker='^', label=f"Test (ρ={r_te:.2f})")
    ax.scatter(x_tr, y_tr, s=18, alpha=0.5, label=f"Train")
    ax.scatter(x_te, y_te, s=24, alpha=0.9, marker='^', label=f"Test")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel);
    # ax.set_title(title)
    if annotate:
        txt = f"All ρ={r_all:.2f}, Train ρ={r_tr:.2f}, Test ρ={r_te:.2f}"
        # ax.text(0.02, 0.98, txt, transform=ax.transAxes,
        #         va='top', ha='left', fontsize=10,
        #         bbox=dict(fc='white', ec='0.8', alpha=0.8))
        ax.set_title(txt)
    # ax.grid(alpha=0.25);
    ax.legend(loc='best')
    plt.tight_layout()

    # —— 保存：默认仍走期刊风格；如需自定义 DPI / 不用期刊保存，可切换 ——
    if use_journal_save and (dpi_override is None):
        savefig_journal(fig, out_path_with_ext, journal, ext=SAVE_EXT)
    else:
        # 不用期刊保存，或单独覆盖 DPI
        kw = dict(bbox_inches='tight')
        if dpi_override is not None:
            kw['dpi'] = dpi_override
        fig.savefig(out_path_with_ext, **kw)

    plt.close(fig)


def spectral_features_batch(X_t,
                            freq_axis=-2, time_axis=-1,
                            hf_frac=0.33, eps=1e-8, agg='mean',
                            # ↓↓↓ 新增：频率轴/单位
                            freq_grid=None,   # 1D array-like, 物理频率刻度 (len=F)，优先生效
                            f_min=None, f_max=None,  # 若未给 freq_grid，则用 [f_min, f_max] 线性生成
                            hf_cut=None,      # 物理阈值（与 hf_frac 二选一），例如 100e3 表示 100 kHz
                            freq_unit='Hz'    # 输出单位：'Hz' | 'kHz' | 'MHz'
                            ):
    import torch, numpy as np
    X = X_t.detach()
    if X.is_sparse:
        X = X.to_dense()
    X = torch.relu(X).float()

    # --- 规范轴 & 置换到 (..., F, T) 再展平为 (N, C, F, T) ---
    nd = X.ndim
    f_ax = (freq_axis + nd) % nd
    t_ax = (time_axis + nd) % nd
    if f_ax == t_ax:
        raise ValueError("freq_axis 与 time_axis 指到同一维。")
    perm = [d for d in range(nd) if d not in (f_ax, t_ax)] + [f_ax, t_ax]
    X = X.permute(perm).contiguous()

    N, F, T = X.shape[0], X.shape[-2], X.shape[-1]
    if X.ndim == 4:
        C = X.shape[1]
    else:
        C = int(np.prod(X.shape[1:-2]))
        X = X.view(N, C, F, T)

    # --- 物理频率轴 f_phys: (1,1,F) ---
    if freq_grid is not None:
        f_vec = torch.as_tensor(freq_grid, dtype=X.dtype, device=X.device).view(1, 1, F)
    elif (f_min is not None) and (f_max is not None):
        f_vec = torch.linspace(float(f_min), float(f_max), steps=F, device=X.device, dtype=X.dtype).view(1,1,F)
    else:
        # 退化为归一化频率（与旧版一致）
        f_vec = torch.linspace(0.0, 1.0, steps=F, device=X.device, dtype=X.dtype).view(1,1,F)

    # print(f_vec)
    # --- 频谱沿时间求和 ---
    P_f  = X.sum(dim=-1) + eps            # (N,C,F)
    Ptot = P_f.sum(dim=-1, keepdim=True) + eps  # (N,C,1)

    # --- HFER 的高频起点索引 k0 ---
    if hf_cut is not None:
        # 在物理频率轴上找阈值位置
        f_cpu = f_vec.view(-1).detach().cpu().numpy()
        k0 = int(np.searchsorted(f_cpu, float(hf_cut), side='left'))
        k0 = max(0, min(F-1, k0))
    else:
        k0 = int((1.0 - hf_frac) * F)
        k0 = max(0, min(F-1, k0))

    high = P_f[..., k0:].sum(dim=-1)                 # (N,C)
    hfer = (high / (Ptot.squeeze(-1))).clamp(0, 1)   # (N,C)  —— 无量纲

    # --- SC/SBW 用物理频率 f_vec 计算 ---
    sc  = ((P_f * f_vec).sum(dim=-1) / (Ptot.squeeze(-1)))      # (N,C)  物理单位
    sbw = torch.sqrt((P_f * (f_vec - sc.unsqueeze(-1))**2).sum(dim=-1) /
                     (Ptot.squeeze(-1)) + eps)                  # (N,C)  物理单位

    # 单位缩放到期望单位
    unit_scale = {'Hz':1.0, 'kHz':1e3, 'MHz':1e6}.get(freq_unit, 1.0)
    sc  = sc  * unit_scale
    sbw = sbw * unit_scale

    # 通道聚合
    if agg == 'mean':
        H  = hfer.mean(dim=1)
        Cn = sc.mean(dim=1)
        BW = sbw.mean(dim=1)
    elif agg == 'median':
        H  = hfer.median(dim=1).values
        Cn = sc.median(dim=1).values
        BW = sbw.median(dim=1).values
    else:
        raise ValueError("agg must be 'mean' or 'median'.")

    feats = torch.stack([H, Cn, BW], dim=1)  # (N,3)
    return feats.cpu().numpy(), ['HFER', 'SC', 'SBW']

# ====================== 分箱（每工况一箱） ======================
def build_group_edges_from_centers(centers, jitter_on_ties=True, eps=1e-8):
    c = np.asarray(centers, float)
    if jitter_on_ties:
        uniq, counts = np.unique(c, return_counts=True)
        if np.any(counts > 1):
            order = np.argsort(c, kind='mergesort')
            ranks = np.empty_like(order); ranks[order] = np.arange(len(c))
            c = c + eps * ranks
    c = np.sort(c)
    if len(c) <= 1:
        return np.array([-np.inf, np.inf], float)
    mids = (c[:-1] + c[1:]) / 2.0
    edges = np.concatenate(([-np.inf], mids, [np.inf]))
    return edges

# ====================== 对比增强（显示变换） ======================
def make_probit_transform(*arrays):
    try:
        from scipy.stats import norm
    except Exception:
        warnings.warn("scipy 不可用，probit 退化为 z-score。")
        concat = np.concatenate([a.ravel() for a in arrays if a.size], axis=0)
        mu, sd = np.mean(concat), np.std(concat)+1e-9
        return lambda x: (x-mu)/sd

    concat = np.concatenate([a.ravel() for a in arrays if a.size], axis=0)
    order = np.argsort(concat, kind='mergesort')
    ranks = np.empty_like(order); ranks[order] = np.arange(len(concat))
    u = (ranks + 0.5) / len(concat)
    z = norm.ppf(u)
    x_sorted = np.sort(concat); z_sorted = np.sort(z)
    def f(x):
        if x.size == 0: return x
        return np.interp(x, x_sorted, z_sorted)
    return f

def make_yeojohnson_transform(*arrays):
    concat = np.concatenate([a.ravel() for a in arrays if a.size], axis=0)[:,None]
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    pt.fit(concat)
    def f(x):
        if x.size == 0: return x
        return pt.transform(x.reshape(-1,1)).ravel()
    return f

def make_delta_baseline_transform(bin_medians, baseline_bin=0, scale='iqr'):
    mb = float(bin_medians[baseline_bin])
    if scale == 'iqr':
        valid = np.array(bin_medians)[~np.isnan(bin_medians)]
        s = np.subtract(*np.percentile(valid, [75,25])) if valid.size else 1.0
        s = s if s > 1e-9 else 1.0
    elif isinstance(scale, (int, float)):
        s = float(scale) if scale != 0 else 1.0
    else:
        s = 1.0
    return lambda x: (x - mb) / s

def choose_contrast_transform(mode, combined_bins, pred_bins):
    all_combined = np.concatenate([b for b in combined_bins if b.size], axis=0) if len(combined_bins)>0 else np.array([])
    all_pred     = np.concatenate([b for b in pred_bins     if b.size], axis=0) if len(pred_bins)>0     else np.array([])
    if mode == 'none':
        return lambda x: x
    if mode == 'probit':
        return make_probit_transform(all_combined, all_pred)
    if mode == 'yeo':
        return make_yeojohnson_transform(all_combined, all_pred)
    if mode == 'delta':
        med = [np.nanmedian(b) if b.size else np.nan for b in combined_bins]
        return make_delta_baseline_transform(med, baseline_bin=0, scale='iqr')
    raise ValueError("contrast mode must be in {'none','probit','yeo','delta'}")

# ====================== 可视化与保存：每分组首样本 (X,Y) ======================
def _robust_rescale(img, p=(2,98)):
    vmin, vmax = np.percentile(img, p)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        mu, sd = float(np.mean(img)), float(np.std(img)+1e-9)
        vmin, vmax = mu - 2*sd, mu + 2*sd
    out = (img - vmin) / (vmax - vmin + 1e-12)
    return np.clip(out, 0, 1)

def save_XY_images(X, Y, out_dir, basename):
    """
    X: (C,H,W) ndarray
    Y: (H,W)    ndarray
    保存：
      - {basename}_X.npy, {basename}_Y.npy
      - 每个通道各一张彩色图：{basename}_ch{1..C}.png（jet）
      - Y 使用 half_cmap（coolwarm 的后半段暖色）
    """
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{basename}_X.npy"), X.astype(np.float32))
    np.save(os.path.join(out_dir, f"{basename}_Y.npy"), Y.astype(np.float32))

    X = X.astype(np.float32)
    finite_mask = np.isfinite(X)
    if not finite_mask.any():
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(X))
        vmax = float(np.nanmax(X))
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            vmin, vmax = np.percentile(X[finite_mask], [2, 98])
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
                vmin, vmax = 0.0, 1.0

    cmap = mpl.cm.get_cmap('jet')
    C = X.shape[0]
    denom = (vmax - vmin + 1e-12)
    for ch in range(C):
        x_norm = ((X[ch] - vmin) / denom).clip(0.0, 1.0)
        rgba = cmap(x_norm)
        rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
        imageio.imwrite(os.path.join(out_dir, f"{basename}_ch{ch + 1}.png"), rgb)

    full_cmap = cm.get_cmap('coolwarm', 256)
    half_cmap = full_cmap(np.linspace(0, 0.5, 128))
    half_cmap = cm.colors.ListedColormap(half_cmap)
    fig2, ax2 = plt.subplots()
    ax2.imshow(_robust_rescale(Y), cmap=half_cmap, interpolation='nearest')
    ax2.set_axis_off()
    plt.tight_layout()
    out_path_y = os.path.join(out_dir, f"{basename}_Y_mask.{SAVE_EXT}")
    savefig_journal(fig2, out_path_y, journal, ext=SAVE_EXT)
    plt.close(fig2)

def get_base_dataset(loader):
    ds = loader.dataset
    return ds.dataset if isinstance(ds, Subset) else ds

def save_group_first_pairs(base_dataset, indices, group_size, out_root, tag):
    if indices is None or len(indices) == 0:
        return
    out_dir = os.path.join(out_root, "group_examples", tag)
    os.makedirs(out_dir, exist_ok=True)
    saved = set()
    for idx in sorted(indices):
        gid = idx // group_size
        if gid in saved:
            continue
        sample = base_dataset[idx]
        try:
            X, Y = unpack_batch(sample)
        except Exception:
            if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                X, Y = sample[0], sample[1]
            elif isinstance(sample, dict):
                X, Y = sample.get('X', sample.get('x')), sample.get('Y', sample.get('y'))
            else:
                raise
        X_np = to_numpy(X)
        if X_np.ndim == 4:
            if X_np.shape[-1] == 1:
                X_np = X_np[...,0]
            elif X_np.shape[0] == 1 and X_np.shape[1] > 1:
                X_np = X_np[0]
        if X_np.ndim != 3:
            raise ValueError(f"Expect X as (C,H,W), got {X_np.shape}")

        Y_np = normalize_y_shape(Y)
        if Y_np.ndim == 3 and Y_np.shape[0] == 1:
            Y_np = Y_np[0]
        if Y_np.ndim != 2:
            raise ValueError(f"Expect Y as (H,W), got {Y_np.shape}")

        gid_dir = os.path.join(out_dir, f"gid_{gid:05d}")
        basename = f"gid_{gid:05d}"
        save_XY_images(X_np, Y_np, gid_dir, basename)
        saved.add(gid)

# ====================== 1) 主图（交换坐标轴版本） ======================
def plot_binned_trend_with_pred_box(
    out_path_with_ext,
    edges,
    order,
    label_centers,
    combined_bins,
    pred_bins,
    xscale='linear',
    yscale='linear',
    contrast_transform=None,
    lowess_frac=0.35,
    orientation='yx',
    use_center_in_label=False
):
    B = len(edges) - 1
    assert len(combined_bins) == B and len(pred_bins) == B
    transf = (lambda x: x) if contrast_transform is None else contrast_transform

    med  = np.array([np.nanmedian(transf(combined_bins[b])) if combined_bins[b].size else np.nan for b in range(B)])
    q25  = np.array([np.nanpercentile(transf(combined_bins[b]), 25) if combined_bins[b].size else np.nan for b in range(B)])
    q75  = np.array([np.nanpercentile(transf(combined_bins[b]), 75) if combined_bins[b].size else np.nan for b in range(B)])

    nonempty = np.array([(combined_bins[b].size > 0) or (pred_bins[b].size > 0) for b in range(B)])
    bin_ids_all = list(range(B))
    if order == 'desc':
        bin_ids_all = bin_ids_all[::-1]
    keep_ids = [b for b in bin_ids_all if nonempty[b]]
    if len(keep_ids) == 0:
        warnings.warn("[plot] all bins are empty; skip plotting.")
        return

    K = len(keep_ids)
    pos = np.arange(K) + 1

    med_keep = np.array([med[b] for b in keep_ids], dtype=float)
    q25_keep = np.array([q25[b] for b in keep_ids], dtype=float)
    q75_keep = np.array([q75[b] for b in keep_ids], dtype=float)

    if use_center_in_label and (label_centers is not None):
        centers_arr = np.asarray(label_centers, float)
        tick_labels = [
            (f"bin{b+1}\n{centers_arr[b]:.2f}" if np.isfinite(centers_arr[b]) else f"bin{b+1}")
            for b in keep_ids
        ]
    else:
        tick_labels = [f"bin{b+1}" for b in keep_ids]

    fig, ax = plt.subplots()

    if orientation == 'yx':
        ax.fill_between(pos, q25_keep, q75_keep, alpha=0.18, edgecolor='none', label='IQR')
        ax.plot(pos, med_keep, '-o', color='#019620', label='Median', lw=journal["line_width"])

        if np.isfinite(med_keep).sum() >= 4:
            xs_s, ys_s = lowess_smooth(pos, med_keep, frac=lowess_frac)
            ax.plot(xs_s, ys_s, linestyle='--', color='#de0611', lw=journal["line_width"], label='Trend reference')

        # pred_positions, pred_data = [], []
        # for i, b in enumerate(keep_ids):
        #     if pred_bins[b].size > 0:
        #         pred_positions.append(pos[i])
        #         pred_data.append(transf(pred_bins[b]))
        # if len(pred_data) > 0:
        #     bp_te = ax.boxplot(pred_data, positions=pred_positions, vert=True, widths=0.35,
        #                        showfliers=False, patch_artist=True)
        #     for patch in bp_te['boxes']:
        #         patch.set(facecolor='#ffccbc', edgecolor='#e64a19', alpha=0.9)
        #     for medline in bp_te['medians']:
        #         medline.set(color='#d84315', linewidth=journal["line_width"])

        if yscale and yscale != 'linear':
            ax.set_yscale(yscale)
        ax.set_xticks(pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_xlabel(r"$s^{A}_{Y}$ (bins)")
        ax.set_ylabel(r"$s_{X}$ (standardized)")
        ax.grid(axis='y', alpha=0.2)
        ax.legend(loc='best')

    else:
        ax.fill_betweenx(pos, q25_keep, q75_keep, alpha=0.18, edgecolor='none', label='IQR')
        ax.plot(med_keep, pos, '-o', color='#37474f', label='Median', lw=journal["line_width"])

        if np.isfinite(med_keep).sum() >= 4:
            ys_s, xs_s = lowess_smooth(pos, med_keep, frac=lowess_frac)
            ax.plot(xs_s, ys_s, linestyle='--', color='#1b5e20', lw=journal["line_width"], label='Trend reference')

        # pred_positions, pred_data = [], []
        # for i, b in enumerate(keep_ids):
        #     if pred_bins[b].size > 0:
        #         pred_positions.append(pos[i])
        #         pred_data.append(transf(pred_bins[b]))
        # if len(pred_data) > 0:
        #     bp_te = ax.boxplot(pred_data, positions=pred_positions, vert=False, widths=0.35,
        #                        showfliers=False, patch_artist=True)
        #     for patch in bp_te['boxes']:
        #         patch.set(facecolor='#ffccbc', edgecolor='#e64a19', alpha=0.9)
        #     for medline in bp_te['medians']:
        #         medline.set(color='#d84315', linewidth=journal["line_width"])

        if yscale and yscale != 'linear':
            ax.set_xscale(yscale)
        ax.set_yticks(pos)
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel(r"$s_{X}$ (standardized)")
        ax.set_ylabel(r"$s^{A}_{Y}$ (bins)")
        ax.grid(axis='x', alpha=0.2)
        ax.legend(loc='best')

    plt.tight_layout()
    savefig_journal(fig, out_path_with_ext, journal, ext=SAVE_EXT)
    plt.close(fig)

# ====================== 2) 误差箱线图 ======================
def y_binned_error_boxplots(
    out_path_with_ext, sY_te_grp, sYhat_te_grp, edges,
    title='|Ŷ − Y| vs sY bins (test, grouped)'
):
    sY_te_grp   = np.asarray(sY_te_grp, float)
    sYhat_te_grp= np.asarray(sYhat_te_grp, float)
    err = np.abs(sYhat_te_grp - sY_te_grp)

    idx = np.clip(np.digitize(sY_te_grp, edges) - 1, 0, len(edges)-2)
    bins = []
    for b in range(len(edges)-1):
        m = (idx == b)
        if m.sum() == 0: continue
        bins.append(dict(b=b, e=err[m]))

    fig, ax = plt.subplots()
    y_pos = np.arange(len(bins)) + 1
    labels = [f"bin{bi+1}" for bi in range(len(bins))]
    data = [b['e'] for b in bins]
    bp = ax.boxplot(data, positions=y_pos, vert=False, widths=0.35, showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set(facecolor='#ffe0b2', edgecolor='#ef6c00', alpha=0.95)
    for med in bp['medians']:
        med.set(color='#e65100', linewidth=journal["line_width"])

    ax.set_yticks(y_pos); ax.set_yticklabels(labels)
    ax.set_xlabel("|Ŷ − Y| (standardized)")
    ax.set_ylabel("sY bins (per-group)")
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    savefig_journal(fig, out_path_with_ext, journal, ext=SAVE_EXT)
    plt.close(fig)

# ====================== KS 近似 ======================
def _ks_distance(a, b):
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).statistic)
    except Exception:
        ax = np.sort(a)
        bx = np.sort(b)
        grid = np.unique(np.concatenate([ax, bx]))
        Fa = np.searchsorted(ax, grid, side='right') / len(ax)
        Fb = np.searchsorted(bx, grid, side='right') / len(bx)
        return float(np.max(np.abs(Fa - Fb)))

# ====================== 评估（采集→全数据拟合→聚合→绘图 + 保存首样本 + 特征散点） ======================
def evaluate_fold_ybinned(
    model, train_loader, val_loader, device, out_dir,
    train_idx, val_idx, group_size=128,
    primary_sy='area', sx_mode='pc1',
    encoder_module_name='encoder',
    order='asc',
    contrast='probit',
    xscale='none',
    yscale='none',
    lowess_frac=0.35,
    save_group_examples=True,
    orientation='yx'
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # 保存每分组首样本
    if save_group_examples:
        base_ds_train = get_base_dataset(train_loader)
        base_ds_test  = get_base_dataset(val_loader)
        save_group_first_pairs(base_ds_train, train_idx, group_size, out_dir, tag='train')
        save_group_first_pairs(base_ds_test,  val_idx,   group_size, out_dir, tag='test')

    # -------- 收集训练（样本级） --------
    feat_tr_list, gid_tr_list = [], []
    Z_tr_list, sY_tr_list, gid_tr_list = [], [], []
    idx_ptr = 0
    with torch.no_grad():
        for batch in train_loader:
            X, Y = unpack_batch(batch)
            bs = X.shape[0]
            X = X.to(device).float(); Y = Y.to(device).float()

            feats_np, feat_names = spectral_features_batch(
                X, freq_axis=-2, time_axis=-1, hf_frac=0.33, agg='mean',
                f_min=200, f_max=4000,    # 0~200 kHz
                hf_cut=120,              # 高于 120 kHz 视为“高频”
                freq_unit='kHz'
            )
            feat_tr_list.append(feats_np)

            z = extract_latent_batch(model, X, module_name=encoder_module_name)
            z_np = to_numpy(z)

            Y_np = normalize_y_shape(Y)
            batch_indices = np.array(train_idx[idx_ptr: idx_ptr+bs]); idx_ptr += bs
            gids = (batch_indices // group_size).astype(int)

            for i in range(bs):
                mets = compute_sy_metrics(Y_np[i])
                sY_tr_list.append(pick_sy(mets, primary_sy))
            Z_tr_list.append(z_np); gid_tr_list.append(gids)

    Z_tr = np.concatenate(Z_tr_list, axis=0)
    sY_tr = np.asarray(sY_tr_list, float)
    F_tr = np.concatenate(feat_tr_list, axis=0)  # (Ntr, 3)
    gid_tr = np.concatenate(gid_tr_list, axis=0)

    # -------- 收集测试（样本级） --------
    feat_te_list, gid_te_list = [], []
    Z_te_list, sY_te_list, sYhat_te_list, gid_te_list = [], [], [], []
    idx_ptr = 0
    with torch.no_grad():
        for batch in val_loader:
            X, Y = unpack_batch(batch)
            bs = X.shape[0]
            X = X.to(device).float(); Y = Y.to(device).float()
            Yhat = model(X)

            feats_np, feat_names = spectral_features_batch(
                X, freq_axis=-2, time_axis=-1, hf_frac=0.33, agg='mean',
                f_min=200, f_max=4000,  # 0~200 kHz
                hf_cut=120,  # 高于 120 kHz 视为“高频”
                freq_unit='kHz'
            )
            feat_te_list.append(feats_np)

            z = extract_latent_batch(model, X, module_name=encoder_module_name)
            z_np = to_numpy(z)

            Y_np = normalize_y_shape(Y)
            Yhat_np = normalize_y_shape(Yhat)
            batch_indices = np.array(val_idx[idx_ptr: idx_ptr+bs]); idx_ptr += bs
            gids = (batch_indices // group_size).astype(int)

            for i in range(bs):
                mets_y  = compute_sy_metrics(Y_np[i])
                mets_yh = compute_sy_metrics(Yhat_np[i])
                sY_te_list.append(pick_sy(mets_y, primary_sy))
                sYhat_te_list.append(pick_sy(mets_yh, primary_sy))
            Z_te_list.append(z_np); gid_te_list.append(gids)

    Z_te = np.concatenate(Z_te_list, axis=0)
    sY_te = np.asarray(sY_te_list, float)
    sYhat_te = np.asarray(sYhat_te_list, float)
    F_te = np.concatenate(feat_te_list, axis=0)  # (Nte, 3)
    gid_te = np.concatenate(gid_te_list, axis=0)

    # -------- 用“整套数据（train+test）”拟合 PCA & 标准化 --------
    Z_all = np.concatenate([Z_tr, Z_te], axis=0)
    sX_all_raw, pca_model = sx_from_z_train(Z_all, mode=sx_mode)
    sX_tr_raw = sX_all_raw[:len(Z_tr)]
    sX_te_raw = sX_all_raw[len(Z_tr):]

    scaler_x = RobustStandardizer().fit(np.concatenate([sX_tr_raw, sX_te_raw], axis=0))
    scaler_y = RobustStandardizer().fit(np.concatenate([sY_tr, sY_te], axis=0))

    sX_tr = scaler_x.transform(sX_tr_raw)
    sX_te = scaler_x.transform(sX_te_raw)
    sY_tr_std = scaler_y.transform(sY_tr)
    sY_te_std = scaler_y.transform(sY_te)
    sYhat_te_std = scaler_y.transform(sYhat_te)

    # -------- 按工况聚合到“工况级”：sY 取中位；sX 保留分布；特征取中位 --------
    grp_to_sY_tr = defaultdict(list); grp_to_sX_tr = defaultdict(list)
    grp_to_F_tr = defaultdict(list)
    for f, g in zip(F_tr, gid_tr):
        grp_to_F_tr[g].append(f)
    feat_tr_grp = []
    for g in sorted(grp_to_F_tr.keys()):
        arr = np.vstack(grp_to_F_tr[g])
        feat_tr_grp.append(np.median(arr, axis=0))
    feat_tr_grp = np.asarray(feat_tr_grp)  # (G_tr, 3)

    for x, y, g in zip(sX_tr, sY_tr_std, gid_tr):
        grp_to_sY_tr[g].append(y); grp_to_sX_tr[g].append(x)

    grp_to_sY_te = defaultdict(list); grp_to_sYh_te = defaultdict(list); grp_to_sX_te = defaultdict(list)
    grp_to_F_te = defaultdict(list)
    for f, g in zip(F_te, gid_te):
        grp_to_F_te[g].append(f)
    feat_te_grp = []
    for g in sorted(grp_to_F_te.keys()):
        arr = np.vstack(grp_to_F_te[g])
        feat_te_grp.append(np.median(arr, axis=0))
    feat_te_grp = np.asarray(feat_te_grp)  # (G_te, 3)

    for x, y, yh, g in zip(sX_te, sY_te_std, sYhat_te_std, gid_te):
        grp_to_sY_te[g].append(y); grp_to_sYh_te[g].append(yh); grp_to_sX_te[g].append(x)

    sY_tr_grp, sX_tr_list = [], []
    for g in sorted(grp_to_sY_tr.keys()):
        sY_tr_grp.append(np.median(grp_to_sY_tr[g]))
        sX_tr_list.append(np.asarray(grp_to_sX_tr[g]))
    sY_te_grp, sYhat_te_grp, sX_te_list = [], [], []
    for g in sorted(grp_to_sY_te.keys()):
        sY_te_grp.append(np.median(grp_to_sY_te[g]))
        sYhat_te_grp.append(np.median(grp_to_sYh_te[g]))
        sX_te_list.append(np.asarray(grp_to_sX_te[g]))

    sY_tr_grp   = np.asarray(sY_tr_grp, float)
    sY_te_grp   = np.asarray(sY_te_grp, float)
    sYhat_te_grp= np.asarray(sYhat_te_grp, float)

    # -------- 用“全部真实工况”的 sY 中心值生成分箱（每工况一箱） --------
    centers_all = np.concatenate([sY_tr_grp, sY_te_grp], axis=0)
    edges_all = build_group_edges_from_centers(centers_all, jitter_on_ties=True)

    # 每箱的标签中心
    centers_sorted = np.sort(centers_all)
    bin_ids_for_centers = np.clip(np.digitize(centers_sorted, edges_all) - 1, 0, len(edges_all)-2)
    label_centers = []
    for b in range(len(edges_all)-1):
        vals = centers_sorted[bin_ids_for_centers == b]
        label_centers.append(np.median(vals) if vals.size else np.nan)
    label_centers = np.array(label_centers, float)

    # -------- 聚合到“真实 bins（基于 Y）” --------
    B = len(edges_all)-1
    combined_bins = [np.array([]) for _ in range(B)]
    idx_tr_bins = np.clip(np.digitize(sY_tr_grp, edges_all) - 1, 0, B-1)
    for i, b in enumerate(idx_tr_bins):
        combined_bins[b] = np.concatenate([combined_bins[b], sX_tr_list[i]]) if combined_bins[b].size else np.asarray(sX_tr_list[i])
    idx_te_bins = np.clip(np.digitize(sY_te_grp, edges_all) - 1, 0, B-1)
    for i, b in enumerate(idx_te_bins):
        combined_bins[b] = np.concatenate([combined_bins[b], sX_te_list[i]]) if combined_bins[b].size else np.asarray(sX_te_list[i])

    pred_bins = [np.array([]) for _ in range(B)]
    idx_pred_bins = np.clip(np.digitize(sYhat_te_grp, edges_all) - 1, 0, B-1)
    for i, b in enumerate(idx_pred_bins):
        pred_bins[b] = np.concatenate([pred_bins[b], sX_te_list[i]]) if pred_bins[b].size else np.asarray(sX_te_list[i])

    # -------- 对比增强（仅显示） --------
    contrast = contrast.lower()
    if contrast not in ('none','probit','yeo','delta'):
        raise ValueError("--contrast must be one of: none, probit, yeo, delta")
    contrast_fn = choose_contrast_transform(contrast, combined_bins, pred_bins)

    # -------- 主图：期刊输出 + 坐标交换 --------
    title_stub = f"{primary_sy}_{sx_mode}_{contrast}_{order}"
    out_main = os.path.join(out_dir, f"trend_IQR_LOWESS_ticktxt_{title_stub}.{SAVE_EXT}")
    plot_binned_trend_with_pred_box(
        out_main,
        edges=edges_all,
        order=order,
        label_centers=label_centers,
        combined_bins=combined_bins,
        pred_bins=pred_bins,
        xscale='linear' if xscale == 'none' else xscale,
        yscale='linear' if yscale == 'none' else yscale,
        contrast_transform=contrast_fn,
        lowess_frac=lowess_frac,
        orientation=orientation
    )

    # -------- 误差箱线图：期刊输出 --------
    out_err = os.path.join(out_dir, f"ybinned_error_{primary_sy}_{sx_mode}.{SAVE_EXT}")
    y_binned_error_boxplots(
        out_err, sY_te_grp=sY_te_grp, sYhat_te_grp=sYhat_te_grp, edges=edges_all,
        title=f"|Ŷ−Y| vs sY (per-group bins) | {primary_sy}"
    )

    # ======== 新增：光谱特征 vs sY、vs sX 的散点 + ρ（工况级） ========

    # sX 工况级中位数（train/test）
    sX_tr_grp_med = np.array([np.median(x) for x in sX_tr_list]) if len(sX_tr_list)>0 else np.array([])
    sX_te_grp_med = np.array([np.median(x) for x in sX_te_list]) if len(sX_te_list)>0 else np.array([])

    # === X 轴单位标签（与 spectral_features_batch(freq_unit=...) 保持一致）===
    freq_unit_for_labels = 'kHz'  # 改成你实际用的单位：'Hz' / 'kHz' / 'MHz'

    FEAT_XLABELS = {
        'HFER': 'High-Freq Energy Ratio of X (–)',  # HFER 无量纲
        'SC': f'SC of X ({freq_unit_for_labels})',
        'SBW': f'SBW of X ({freq_unit_for_labels})',
    }

    fnames = ['HFER', 'SC', 'SBW']

    # 1) 特征 vs sY（工况级；y 轴是标准化后的 sY）
    for j, name in enumerate(fnames):
        outp = os.path.join(out_dir, f"feat_{name}_vs_sY_GROUP_{primary_sy}.{SAVE_EXT}")
        scatter_with_rho(
            x_tr=(feat_tr_grp[:, j] if feat_tr_grp.size else np.array([])),
            y_tr=sY_tr_grp,
            x_te=(feat_te_grp[:, j] if feat_te_grp.size else np.array([])),
            y_te=sY_te_grp,
            xlabel=FEAT_XLABELS[name],
            ylabel=r"$s_{Y}$ (standardized)",
            out_path_with_ext=outp,
            title=f"{name} vs sY (group-level)",
            fig_size_cm=(12, 11)  # ← 可自定义图窗大小（厘米）；或用 fig_size=(5,3.6)
        )

    # 2) 特征 vs sX（工况级；y 轴用 sX 工况中位，已标准化）
    for j, name in enumerate(fnames):
        outp = os.path.join(out_dir, f"feat_{name}_vs_sX_GROUP_{primary_sy}.{SAVE_EXT}")
        scatter_with_rho(
            x_tr=(feat_tr_grp[:, j] if feat_tr_grp.size else np.array([])),
            y_tr=sX_tr_grp_med,
            x_te=(feat_te_grp[:, j] if feat_te_grp.size else np.array([])),
            y_te=sX_te_grp_med,
            xlabel=FEAT_XLABELS[name],
            ylabel="$s_{X}$ (standardized)",
            out_path_with_ext=outp,
            title=f"{name} vs sX (group-level)",
            fig_size_cm=(12, 11)
        )

    # -------- 指标 --------
    def cliff_delta(a,b, seed=0, maxn=5000):
        a = np.asarray(a); b = np.asarray(b)
        if a.size==0 or b.size==0: return np.nan
        rng = np.random.default_rng(seed)
        aa = a if a.size<=maxn else rng.choice(a, maxn, replace=False)
        bb = b if b.size<=maxn else rng.choice(b, maxn, replace=False)
        cmp = aa[:,None] - bb[None,:]
        wins = (cmp > 0).mean(); ties = (cmp == 0).mean()
        return float(2*wins + ties - 1.0)

    sX_tr_all = np.concatenate(sX_tr_list, axis=0) if len(sX_tr_list)>0 else np.array([])
    sX_te_all = np.concatenate(sX_te_list, axis=0) if len(sX_te_list)>0 else np.array([])

    # 额外统计：三种特征与 sY/sX 的 Spearman（合并 train+test 的工况级）
    feat_metrics = {}
    for j, name in enumerate(fnames):
        x_all_sy = np.concatenate([feat_tr_grp[:, j], feat_te_grp[:, j]]) if (feat_tr_grp.size and feat_te_grp.size) else np.array([])
        y_all_sy = np.concatenate([sY_tr_grp, sY_te_grp])
        rho_feat_sy = spearman_rho(x_all_sy, y_all_sy) if x_all_sy.size else np.nan

        x_all_sx = x_all_sy  # 同一特征
        y_all_sx = np.concatenate([sX_tr_grp_med, sX_te_grp_med])
        rho_feat_sx = spearman_rho(x_all_sx, y_all_sx) if x_all_sx.size else np.nan

        feat_metrics[f"{name}_rho_vs_sY_group"] = float(rho_feat_sy)
        feat_metrics[f"{name}_rho_vs_sX_group"] = float(rho_feat_sx)

    metrics = dict(
        spearman_train=spearman_rho(sX_tr, sY_tr),
        spearman_test=spearman_rho(sX_te, sY_te),
        ks_shift=_ks_distance(sX_tr, sX_te),
        coverage=float(np.mean((sX_te >= np.percentile(sX_tr, 2.5)) &
                               (sX_te <= np.percentile(sX_tr, 97.5)))),
        med_abs_err=float(np.median(np.abs(sYhat_te - sY_te))),
    )
    metrics.update(feat_metrics)

    return metrics

# ====================== 主流程 ======================
def main(pt_dir, param_dir, opt, batch_size, num_workers, device, k_folds,
         encoder_module_name='encoder', sx_mode='pc1', group_size=128,
         order='asc', contrast='probit', xscale='none', yscale='none', lowess_frac=0.35,
         save_group_examples=True, orientation='yx'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ===== TODO: 替换为你项目内的实际导入 =====
    from dataset_pt import PreprocessedPTDataset
    from model.model_cnn_151 import SimpleCNN
    # ==========================================

    dataset = PreprocessedPTDataset(pt_dir)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples.")

    groups = np.arange(N) // group_size
    gkf = GroupKFold(n_splits=k_folds)

    all_metrics = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(np.arange(N), groups=groups)):
        print(f"\n=== Fold {fold + 1}/{k_folds} (grouped by {group_size}) ===")
        fold_dir = os.path.join(param_dir, f"fold_{fold + 1}")

        ckpt_candidates = [
            os.path.join(fold_dir, f"{opt}_PTModel_fold{fold + 1}_epoch500.pth"),
        ]
        model_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)
        if model_path is None:
            print(f"[WARNING] No model checkpoint found in {fold_dir}")
            continue
        print(f"Using checkpoint: {model_path}")

        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

        model = SimpleCNN(opt=opt, drop_path_rate=0.2).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        for primary_sy in ['area','compactness','rsi']:
            out_dir = os.path.join(fold_dir, f"aa0911_ybins_trend_pub_{primary_sy}_{sx_mode}_{contrast}_{order}")
            os.makedirs(out_dir, exist_ok=True)
            mets = evaluate_fold_ybinned(
                model, train_loader, val_loader, device, out_dir,
                train_idx=train_idx, val_idx=val_idx, group_size=group_size,
                primary_sy=primary_sy, sx_mode=sx_mode,
                encoder_module_name=encoder_module_name,
                order=order, contrast=contrast, xscale=xscale, yscale=yscale, lowess_frac=lowess_frac,
                save_group_examples=save_group_examples, orientation=orientation
            )
            print(f"[{primary_sy}] metrics: {mets}")
            for k,v in mets.items():
                all_metrics[(primary_sy,k)].append(v)

    # 汇总
    print("\n=== Summary ===")
    for key, vals in all_metrics.items():
        m = np.nanmean(vals); s = np.nanstd(vals)
        print(f"{key}: mean={m:.4f}  std={s:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\thinckness_binary_128')
    parser.add_argument('--opt', type=str, default='EPS', help='Target type: EPS, Ek, damageM')
    parser.add_argument('--param_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\thickness_binary_128')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--k_folds', type=int, default=8)
    parser.add_argument('--encoder_module_name', type=str, default='encoder',
        help="Module name to hook for latent z. Fallback: last Conv2d + GAP.")
    parser.add_argument('--sx_mode', type=str, default='pc1', choices=['pc1','l2'],
        help="sX from z: 'pc1' (recommend) or 'l2' (norm).")
    parser.add_argument('--group_size', type=int, default=128,
        help='Samples per operating condition (no shuffle during dataset build).')

    parser.add_argument('--order', type=str, default='asc', choices=['asc','desc'],
        help="Display order of sY bins on X-axis.")
    parser.add_argument('--contrast', type=str, default='probit', choices=['none','probit','yeo','delta'],
        help="Display-only contrast enhancement for sX.")
    parser.add_argument('--xscale', type=str, default='none', choices=['none','linear'],
        help="Scale for X-axis (bins).")
    parser.add_argument('--yscale', type=str, default='none', choices=['none','linear','symlog'],
        help="Scale for Y-axis (sX).")
    parser.add_argument('--lowess_frac', type=float, default=0.35,
        help="LOWESS smoothing fraction for reference curve.")
    parser.add_argument('--save_group_examples', default=False, action='store_true',
        help="Save the first (X,Y) pair for each group into npy and images.")
    parser.add_argument('--orientation', type=str, default='yx', choices=['yx', 'xy'],
                        help="yx: X=bin, Y=sX（交换后，当前默认）; xy: X=sX, Y=bin（原横向风格）")

    args = parser.parse_args()
    main(**vars(args))

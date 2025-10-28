#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
depth_diameter_spline_vs_power.py
--------------------------------------------------
1. 读取所有 *_pits.csv
2. Z-score 双侧过滤 (可关)
3. 幂律 (HuberRegressor，robust) 拟合
4. B 样条回归 (三次样条，6 个结点) 拟合
5. 打印 R²，保存同图对比 curves
--------------------------------------------------
"""

import os, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# ========= 用户可配置 =========
ANALYSIS_DIR  = r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\out\analysis'
CSV_PATTERN   = '*6_pits.csv'   # 读取全部试样
PIXEL2MM      = 160/1000            # px→mm；若保持 px 设置 1
Z_TH          = 3              # None 可取消 Z-score 过滤
N_KNOTS       = 8              # B 样条结点数
SHOW_SCATTER  = True           # False → 不画散点
OUT_FIG       = '6_depth_vs_area_spline_power.png'
# ==================================

# ---------- 1. 读取所有 pits ----------
paths = glob.glob(os.path.join(ANALYSIS_DIR, CSV_PATTERN))
if not paths:
    raise RuntimeError(f'No {CSV_PATTERN} found in {ANALYSIS_DIR}')

df_all = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
if not {'area_px2', 'depth_est'} <= set(df_all.columns):
    raise KeyError('pits.csv 缺少 diameter 或 depth_est 列')

x_raw = df_all['area_px2'].values * PIXEL2MM**2  # (N,)
y_raw = df_all['depth_est'].values

# ---------- 2. 基本过滤 ----------
m_valid = np.isfinite(x_raw) & np.isfinite(y_raw) & (x_raw > 0) & (y_raw > 0)
x, y = x_raw[m_valid], y_raw[m_valid]

if Z_TH is not None:
    mz = (np.abs(zscore(x)) < Z_TH) & (np.abs(zscore(y)) < Z_TH)
    print(f'[INFO] Z-score {Z_TH}σ: removed {(~mz).sum()} / {len(mz)} points')
    x, y = x[mz], y[mz]

x2d = x.reshape(-1, 1)       # sklearn 2-D

# ---------- 3. 幂律拟合 (log–log, Huber) ----------
logx, logy = np.log(x).reshape(-1, 1), np.log(y)
lm_pow = LinearRegression()
lm_pow.fit(logx, logy)
logy_pred = lm_pow.predict(logx)
r2_pow = r2_score(logy, logy_pred)
d = lm_pow.coef_[0]              # 指数
c = math.exp(lm_pow.intercept_)  # 系数
print(f'Power   R² = {r2_pow:.4f}   depth = {c:.4g}·diam^{d:.4g}')

# ---------- 4. B 样条回归 ----------
spl = SplineTransformer(degree=3, n_knots=N_KNOTS, include_bias=False)
spline_model = make_pipeline(spl, LinearRegression())
spline_model.fit(x2d, y)
y_pred_spl = spline_model.predict(x2d)
r2_spl = r2_score(y, y_pred_spl)
print(f'Spline  R² = {r2_spl:.4f}   (degree=3, knots={N_KNOTS})')

# ---------- 5. 绘图 ----------
plt.figure(figsize=(6,4))
if SHOW_SCATTER:
    plt.scatter(x, y, s=12, alpha=0.25, label='data')

# 拟合曲线（平滑绘制）
x_line = np.linspace(x.min(), x.max(), 400)
plt.plot(x_line, c * x_line**d,
         lw=2, ls='--', label=f'Power  (R²={r2_pow:.3f})')
# plt.plot(x_line, spline_model.predict(x_line.reshape(-1,1)),
#          lw=2, label=f'Spline (R²={r2_spl:.3f})')

unit = 'mm' if PIXEL2MM != 1 else 'px'
plt.xlabel(f'area ({unit}²)')
plt.ylabel('Depth_est (mm)')
# plt.title('Depth vs Diameter – Spline vs Power')
plt.legend()
plt.grid(True)
plt.tight_layout()
fig_path = os.path.join(ANALYSIS_DIR, OUT_FIG)
plt.savefig(fig_path, dpi=300)
plt.close()
print(f'[INFO] 图已保存: {fig_path}')

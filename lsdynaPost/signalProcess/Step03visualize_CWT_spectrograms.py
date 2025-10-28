import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ========= Journal style =========
journal = {
    "fig_size": (14/2.54, 9/2.54),  # cm -> inch
    "font_name": "Arial",  # 或 "Arial"
    "font_size": 15,
    "line_width": 1.2,
    "axis_line_width": 1.0,
    "dpi": 1200
}
SAVE_EXT = "tiff"  # 可改 "pdf"/"png"/"svg"/"tiff"

def apply_journal_style(j):
    mpl.rcParams.update({
        # 尺寸/分辨率
        "figure.figsize": j["fig_size"],
        "savefig.dpi": j["dpi"],
        # 字体与字号
        "font.family": j["font_name"],
        "font.size": j["font_size"],
        "axes.labelsize": j["font_size"],
        "axes.titlesize": j["font_size"],
        "xtick.labelsize": j["font_size"],
        "ytick.labelsize": j["font_size"],
        "legend.fontsize": j["font_size"],
        # 线宽与轴样式
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
        "axes.grid": False,
        # 数学文本/嵌入字体
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def savefig_journal(fig, path, j, ext="tiff"):
    ext = ext.lower()
    kwargs = dict(dpi=j["dpi"], bbox_inches="tight")
    if ext in ("tif", "tiff"):
        kwargs.update(format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path, **kwargs)

apply_journal_style(journal)
# =================================

# 输入CWT频谱图数据文件夹路径
cwt_data_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_nodoutData_CWTspectrum_npyFiles_1e8\acc'  # 修改为实际路径


# 可视化函数
def visualize_cwt_spectrogram(cwt_data, signal_idx, spectrogram_idx):
    """
    可视化指定信号的CWT频谱图。

    参数:
    cwt_data -- CWT频谱图数据，形状为 [128, 500, 500]
    signal_idx -- 信号索引，0-127，表示选择第几个信号
    spectrogram_idx -- CWT频谱图的频谱索引，0-499，表示选择某个频谱图的切片
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(cwt_data[signal_idx, :, :], aspect='auto', cmap='jet', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.title(f'CWT Spectrum of Signal {signal_idx + 1} (Frequency slice {spectrogram_idx})')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.show()

def visualize_save_cwt_spectrogram(
    cwt_data,
    signal_idx,
    spectrogram_idx,
    out_dir=r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\cwt信号可视化',
    pixel_dt=0.01,                   # 横轴步长（与你原代码一致）
    cmap='jet',                      # 期刊建议可换 'viridis'/'magma'
    jconf=journal,
    save_ext=SAVE_EXT,
    dpi=None                         # 若传 None 使用 jconf["dpi"]
):
    """
    可视化指定信号的CWT频谱图，并按期刊规范保存图像。
    cwt_data: [num_signals, num_scales, num_times]
    """
    os.makedirs(out_dir, exist_ok=True)
    data = cwt_data[signal_idx, :, :]         # [scales, times]
    x = np.arange(data.shape[1]) * pixel_dt   # 横轴 (e.g., 微秒)

    fig, ax = plt.subplots(figsize=jconf["fig_size"])
    im = ax.imshow(
        data,
        aspect='auto',
        cmap=cmap,
        origin='lower',
        extent=[x[0], x[-1], 0, data.shape[0] - 1]
    )
    ax.set_xlabel('t (μs)')   # μs: 用 MathText 更稳妥
    ax.set_ylabel('Scale')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True)

    # 精简的期刊风格 colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label('Amplitude')  # 如果需要标注颜色条含义，打开此行

    fig.tight_layout()
    out_path = os.path.join(out_dir, f'cwt_signal_{signal_idx + 1}_slice_{spectrogram_idx}.{save_ext}')
    # 若手动覆盖 DPI
    if dpi is not None:
        tmp = jconf.copy(); tmp["dpi"] = dpi
        savefig_journal(fig, out_path, tmp, ext=save_ext)
    else:
        savefig_journal(fig, out_path, jconf, ext=save_ext)
    plt.close(fig)


# 遍历CWT频谱图数据文件夹中的所有 .npy 文件
for root, _, files in os.walk(cwt_data_path):
    for file in files:
        if file.endswith('_az_CWTspectrum.npy'):
            print(f"正在加载文件: {file}")
            file_path = os.path.join(root, file)

            try:
                # 加载CWT频谱图数据
                cwt_data = np.load(file_path)
                print(f"成功加载 {file}, 形状: {cwt_data.shape}")

                # 检查数据是否为 [128, 500, 500]
                # if cwt_data.shape != (128, 500, 500):
                #     print(f"警告: {file} 的形状不是 [128, 500, 500]，跳过该文件")
                #     continue

                for i in range(64):  # 0 到 64
                    visualize_save_cwt_spectrogram(
                        cwt_data,
                        signal_idx=i + 64,
                        spectrogram_idx=100,
                        out_dir=r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\cwt信号可视化',
                        pixel_dt=0.01,
                        cmap='jet',  # 如需色觉友好可改 'viridis'
                        jconf=journal,
                        save_ext=SAVE_EXT,
                        dpi=600  # 若想覆盖 jconf 的 1200 DPI
                    )

                # # 可视化某个信号的CWT频谱图
                # for signal_idx in range(128):  # 选择信号的索引，0-127
                #     spectrogram_idx = 250  # 选择某个频谱图的频率切片，0-499
                #
                #     # 可视化第一个信号的某个频谱图切片
                #     visualize_cwt_spectrogram(cwt_data, signal_idx, spectrogram_idx)

            except Exception as e:
                print(f"加载或可视化文件 {file} 时出错: {e}")

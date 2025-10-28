import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_one_topbottom(np_file):
    """
    可视化单个 [N, 2, H, W] 文件
    - data[i, 0, :, :] 表示第 i 个旋转后的 top 图像
    - data[i, 1, :, :] 表示第 i 个旋转后的 bottom 图像
    """
    if not os.path.isfile(np_file):
        print(f"[Error] File not found: {np_file}")
        return

    data = np.load(np_file)
    print(f"[Info] Loaded: {np_file}, shape: {data.shape}")
    print(np.min(data), np.max(data))

    # if len(data.shape) != 4 or data.shape[1] != 2:
    #     print(f"[Warning] Data shape unexpected: expected [N, 2, H, W], got {data.shape}")
    #     return

    N = 1
    H = data.shape[2]
    W = data.shape[3]

    grid_cols = 1
    grid_rows = int(np.ceil(N / grid_cols))

    # -------- 可视化 top 图像 --------
    fig_top, axes_top = plt.subplots(grid_rows, grid_cols, figsize=(5, 2 * grid_rows))
    fig_top.suptitle(f"TOP Channel - {os.path.basename(np_file)}")

    for i in range(N):
        row, col = divmod(i, grid_cols)
        ax = axes_top[row][col] if grid_rows > 1 else axes_top[col]
        ax.imshow(data[i, 0], cmap='jet', origin='lower')
        ax.set_title(f"Rot #{i}")
        ax.axis('off')

    # -------- 可视化 bottom 图像 --------
    fig_bot, axes_bot = plt.subplots(grid_rows, grid_cols, figsize=(16, 2 * grid_rows))
    fig_bot.suptitle(f"BOTTOM Channel - {os.path.basename(np_file)}")

    for i in range(N):
        row, col = divmod(i, grid_cols)
        ax = axes_bot[row][col] if grid_rows > 1 else axes_bot[col]
        ax.imshow(data[i, 1], cmap='jet', origin='lower')
        ax.set_title(f"Rot #{i}")
        ax.axis('off')

    # -------- 可视化 bottom 图像 --------
    fig_bot, axes_bot = plt.subplots(grid_rows, grid_cols, figsize=(5, 2 * grid_rows))
    fig_bot.suptitle(f"BOTTOM-TOP Channel - {os.path.basename(np_file)}")

    # 把 axes 展平成一维数组，便于统一索引
    axes_flat = axes_bot.flatten() if hasattr(axes_bot, 'flatten') else [axes_bot]

    for i in range(N):
        ax = axes_flat[i]
        save_p = os.path.join( r"D:\PHDstudent\博士论文\会议\新建文件夹", f"{os.path.splitext(os.path.basename(np_file))[0]}_{i}.png")
        plt.imsave(save_p, data[i, 1] - data[i, 0], dpi=1200, cmap='coolwarm', format='png')
        # 显示 bottom-top 差值图
        ax.imshow(data[i, 1] - data[i, 0], cmap='coolwarm', origin='lower')
        ax.set_title(f"Rot #{i}")
        ax.axis('off')


    # 如果子图数少于 grid_rows*grid_cols，隐藏多余的 axes
    for j in range(N, len(axes_flat)):
        axes_flat[j].axis('off')

    # 保存整个 Figure
    save_dir = r"D:\PHDstudent\博士论文\会议\新建文件夹"
    os.makedirs(save_dir, exist_ok=True)
    save_t = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(np_file))[0]}.png")

    # 使用 savefig 而不是 imsave
    # fig_bot.savefig(save_t, dpi=600, bbox_inches='tight', pad_inches=0)
    # plt.close(fig_bot)

    plt.tight_layout()
    plt.show()


def batch_visualize_topbottom(np_file_dir, np_file_suffix):
    """
    批量可视化 np_file_dir 目录下所有以指定后缀结尾的 topbottom_R*.npy 文件
    """
    if not os.path.isdir(np_file_dir):
        print(f"[Error] {np_file_dir} is not a valid directory.")
        return

    files = sorted([
        f for f in os.listdir(np_file_dir)
        if f.endswith(np_file_suffix)
    ])

    if len(files) == 0:
        print(f"[Warning] No matching '*{np_file_suffix}' files found.")
        return

    print(f"[Info] Found {len(files)} files for visualization.\n")

    for fname in files:
        full_path = os.path.join(np_file_dir, fname)
        # print(full_path)
        visualize_one_topbottom(full_path)


if __name__ == "__main__":
    # ========== 示例路径，请按需修改 ==========
    np_file_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DamageMorphology"
    np_file_suffix = "_damageM_R512.npy"
    np_file_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_thickness_128"
    np_file_suffix = "_thickness_R128.npy"
    # np_file_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\damageM_Data\NP"
    # np_file_suffix = "_damageM.npy"

    batch_visualize_topbottom(np_file_dir, np_file_suffix)

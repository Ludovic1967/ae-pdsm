import os
import numpy as np
import torch
from torch.utils.data import Dataset

def extract_key(filename):
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    if len(parts) < 3:
        raise ValueError(f"文件名 {filename} 无法解析出 Bumper/ dp/ vp 等信息")
    return '_'.join(parts[:3])

class CombinedDataset(Dataset):
    def __init__(self, signals_dir, images_dir, Ek_dir, eps_dir, opt='damageM', transform=None, global_stats_file=None):
        self.signals_dir = signals_dir
        self.images_dir = images_dir
        self.Ek_dir = Ek_dir
        self.eps_dir = eps_dir
        self.opt = opt
        self.transform = transform

        if global_stats_file is None:
            global_stats_file = f'params/global_stats_{self.opt}.npy'

        signal_files = sorted([f for f in os.listdir(signals_dir) if f.endswith('.npy')])
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        Ek_files = sorted([f for f in os.listdir(Ek_dir) if f.endswith('.npy')])
        eps_files = sorted([f for f in os.listdir(eps_dir) if f.endswith('.npy')])

        self.signal_dict = self._group_files(signal_files)
        self.image_dict = self._group_files(image_files)
        self.Ek_dict = self._group_files(Ek_files)
        self.eps_dict = self._group_files(eps_files)

        if self.opt == 'EPS':
            target_dict = self.eps_dict
        elif self.opt == 'Ek':
            target_dict = self.Ek_dict
        elif self.opt == 'damageM':
            target_dict = self.image_dict
        else:
            raise ValueError(f"Unsupported opt: {self.opt}")

        common_keys = set(self.signal_dict.keys()).intersection(target_dict.keys())
        if not common_keys:
            raise ValueError(f"No matching files found for opt={self.opt} between signals and target directory.")

        self.samples = self._create_samples(common_keys)

        if os.path.exists(global_stats_file):
            print(f"加载保存的全局统计量: {global_stats_file}")
            self.global_stats = np.load(global_stats_file, allow_pickle=True).item()
            print(self.global_stats)
        else:
            print("计算全局统计量...")
            self.global_stats = self._compute_global_stats()
            print(self.global_stats)
            # np.save(global_stats_file, self.global_stats)

    def _group_files(self, file_list):
        file_dict = {}
        for f in file_list:
            key = extract_key(f)
            file_dict.setdefault(key, []).append(f)
        return file_dict

    def _create_samples(self, common_keys):
        samples = []
        for key in sorted(common_keys):
            signal_files = self.signal_dict[key]
            if self.opt == 'EPS':
                target_files = self.eps_dict[key]
            elif self.opt == 'Ek':
                target_files = self.Ek_dict[key]
            elif self.opt == 'damageM':
                target_files = self.image_dict[key]

            for s_file, t_file in zip(signal_files, target_files):
                s_data = np.load(os.path.join(self.signals_dir, s_file), mmap_mode='r')
                t_path = {
                    'EPS': os.path.join(self.eps_dir, t_file),
                    'Ek': os.path.join(self.Ek_dir, t_file),
                    'damageM': os.path.join(self.images_dir, t_file)
                }[self.opt]
                t_data = np.load(t_path, mmap_mode='r')

                n_common = min(s_data.shape[0], t_data.shape[0])
                for idx_infile in range(n_common):
                    samples.append((s_file, t_file, idx_infile))
        return samples

    # def _compute_global_stats(self):
    #     print("[CombinedDataset] 统计 signal 全局 min/max 与 target 按通道 min/max ...")
    #     gmin_signal, gmax_signal = float('inf'), float('-inf')
    #     min_c, max_c = None, None
    #     seen = set()
    #
    #     for s_file, t_file, _ in self.samples:
    #         key = (s_file, t_file)
    #         if key in seen:
    #             continue
    #         seen.add(key)
    #
    #         s_data = np.load(os.path.join(self.signals_dir, s_file), mmap_mode='r')
    #         t_data = np.load({
    #             'EPS': os.path.join(self.eps_dir, t_file),
    #             'Ek': os.path.join(self.Ek_dir, t_file),
    #             'damageM': os.path.join(self.images_dir, t_file)
    #         }[self.opt], mmap_mode='r')
    #
    #         # ---------- 若 target 只有 3 维，自动补一个通道维 ----------
    #         if t_data.ndim == 3:  # 形如 [N, H, W]
    #             t_data = t_data[:, np.newaxis, :, :]  # 变成 [N, 1, H, W]
    #         # print(t_data.shape)
    #
    #         gmin_signal = min(gmin_signal, s_data.min())
    #         gmax_signal = max(gmax_signal, s_data.max())
    #
    #         # target: [N, C, H, W] → min/max over (N,H,W), each channel
    #         c_min = np.min(t_data, axis=(0, 2, 3))  # shape [C]
    #         c_max = np.max(t_data, axis=(0, 2, 3))  # shape [C]
    #         # c_min = np.percentile(t_data, 0.5, axis=(0, 2, 3))  # 第1百分位
    #         # c_max = np.percentile(t_data, 99.5, axis=(0, 2, 3))  # 第99百分位
    #
    #         if min_c is None:
    #             min_c, max_c = c_min, c_max
    #         else:
    #             min_c = np.minimum(min_c, c_min)
    #             max_c = np.maximum(max_c, c_max)
    #
    #     return {
    #         'signal_min': gmin_signal,
    #         'signal_max': gmax_signal,
    #         'target_min_per_channel': min_c,
    #         'target_max_per_channel': max_c
    #     }
    def _compute_global_stats(self):
        print("[CombinedDataset] 统计 signal 和 target 的 min/max 中位数 ...")
        signal_mins, signal_maxs = [], []
        target_mins_list, target_maxs_list = [], []
        seen = set()

        for s_file, t_file, _ in self.samples:
            key = (s_file, t_file)
            if key in seen:
                continue
            seen.add(key)

            s_data = np.load(os.path.join(self.signals_dir, s_file), mmap_mode='r')
            t_data = np.load({
                                 'EPS': os.path.join(self.eps_dir, t_file),
                                 'Ek': os.path.join(self.Ek_dir, t_file),
                                 'damageM': os.path.join(self.images_dir, t_file)
                             }[self.opt], mmap_mode='r')

            signal_mins.append(s_data.min())
            signal_maxs.append(s_data.max())

            if t_data.ndim == 3:  # [N, H, W]
                t_data = t_data[:, np.newaxis, :, :]  # [N, 1, H, W]

            # 每个通道分别收集 min/max
            c_min = np.min(t_data, axis=(0, 2, 3))  # shape [C]
            c_max = np.max(t_data, axis=(0, 2, 3))  # shape [C]
            target_mins_list.append(c_min)
            target_maxs_list.append(c_max)

        # 信号归一化最值：取 min/max 的中位数
        gmin_signal = float(np.median(signal_mins))
        gmax_signal = float(np.median(signal_maxs))

        # target 每通道归一化最值：取中位数
        all_target_mins = np.stack(target_mins_list, axis=0)  # [num_samples, C]
        all_target_maxs = np.stack(target_maxs_list, axis=0)  # [num_samples, C]
        min_c = np.max(all_target_mins, axis=0)*1.001  # shape [C]
        max_c = np.median(all_target_maxs, axis=0)  # shape [C]

        return {
            'signal_min': gmin_signal,
            'signal_max': gmax_signal,
            'target_min_per_channel': min_c,
            'target_max_per_channel': max_c
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_file, t_file, idx_infile = self.samples[idx]

        # ----------- signal ------------
        s_data = np.load(os.path.join(self.signals_dir, s_file))
        signal = s_data[idx_infile]  # shape: [...]
        signal = (signal - self.global_stats['signal_min']) / (
                self.global_stats['signal_max'] - self.global_stats['signal_min'] + 1e-12)

        # ----------- target ------------
        if self.opt == 'EPS':
            t_path = os.path.join(self.eps_dir, t_file)
        elif self.opt == 'Ek':
            t_path = os.path.join(self.Ek_dir, t_file)
        elif self.opt == 'damageM':
            t_path = os.path.join(self.images_dir, t_file)

        target = np.load(t_path)[idx_infile]  # 可能是 [C,H,W] 也可能是 [H,W]

        # ----------- 通道独立归一化 ------------
        min_c = self.global_stats['target_min_per_channel'][:target.shape[0]][..., None, None]
        max_c = self.global_stats['target_max_per_channel'][:target.shape[0]][..., None, None]

        target = (target - min_c) / (max_c - min_c + 1e-12)
        target = np.clip(target, 0.0, 10.0).astype(np.float32)
        # print(np.max(target),np.min(target))

        return torch.from_numpy(signal).float(), torch.from_numpy(target).float()


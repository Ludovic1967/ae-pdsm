import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from CombinedDataset import CombinedDataset

def preprocess_and_save_pt(signals_dir, damageM_dir, Ek_dir, eps_dir, opt, output_dir, num_workers=8):
    # 自动匹配对应 opt 的 global_stats 文件
    global_stats_file = f'params/global_stats_{opt}.npy'
    dataset = CombinedDataset(signals_dir, damageM_dir, Ek_dir, eps_dir,
                               opt=opt, global_stats_file=global_stats_file)

    os.makedirs(output_dir, exist_ok=True)

    # 设置DataLoader，使用多进程加载数据
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 用多进程写入预处理数据
    def save_sample(idx, signal, target):
        # 检查是否是5D并去掉多余的维度
        # print(signal)
        if signal.dim() == 4 and signal.shape[0] == 1:  # [1, 4, H, W]
            signal = signal.squeeze(0)  # [4, H, W]
        if target.dim() == 4 and target.shape[0] == 1:  # [1, C, H, W]
            target = target.squeeze(0)  # [C, H, W]
        torch.save((signal, target), os.path.join(output_dir, f'sample_{idx:05d}.pt'))

    for idx, (signal, target) in tqdm(enumerate(data_loader), desc=f"Preprocessing {opt}", total=len(dataset)):
        save_sample(idx, signal, target)  # 批量写入

if __name__ == '__main__':
    # preprocess_and_save_pt(
    #     signals_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT_512',
    #     damageM_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DamageMorphology',
    #     Ek_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DebrisCloudEk',
    #     eps_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain',
    #     opt='EPS',  # 或 'Ek', 'damageM'
    #     output_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\EPS_test',
    #     num_workers=16  # 适当调整这个值以匹配你的CPU核心数量
    # )

    # preprocess_and_save_pt(
    #     signals_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT_512',
    #     damageM_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DamageMorphology',
    #     Ek_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DebrisCloudEk',
    #     eps_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain',
    #     opt='damageM',  # 或 'Ek', 'damageM'
    #     output_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\damageM',
    #     num_workers=16  # 适当调整这个值以匹配你的CPU核心数量
    # )

    # preprocess_and_save_pt(
    #     signals_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT_512',
    #     damageM_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DamageMorphology',
    #     Ek_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DebrisCloudEk',
    #     eps_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain',
    #     opt='Ek',  # 或 'Ek', 'damageM'
    #     output_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\Ek',
    #     num_workers=16  # 适当调整这个值以匹配你的CPU核心数量
    # )

    preprocess_and_save_pt(
        signals_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT_128',
        damageM_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_thickness500_binary_128',
        Ek_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DebrisCloudEk',
        eps_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain',
        opt='damageM',  # 或 'Ek', 'damageM'
        output_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\thinckness500_binary_128',
        num_workers=16  # 适当调整这个值以匹配你的CPU核心数量
    )  #train/valid database

    # preprocess_and_save_pt(
    #     signals_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\eval\DataBase_Signal_CWT_512',
    #     damageM_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\eval\DataBase_thickness',
    #     Ek_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DebrisCloudEk',
    #     eps_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain',
    #     opt='damageM',  # 或 'Ek', 'damageM'
    #     output_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\eval\PreprocessedPT\thickness',
    #     num_workers=16  # 适当调整这个值以匹配你的CPU核心数量
    # )  #test database

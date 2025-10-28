import os
import shutil

# 设置路径
reference_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\damageM_Data\P'  # 参考文件所在路径
# target_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DebrisCloudEk'        # 需要处理的文件路径
# output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DebrisCloudEk\P'       # 存放匹配文件的新建文件夹

target_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain'        # 需要处理的文件路径
output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain\P'       # 存放匹配文件的新建文件夹

# 创建输出文件夹（如果不存在）
os.makedirs(output_path, exist_ok=True)

# 读取参考文件名，并提取前三段作为匹配关键字
reference_keys = set()
for fname in os.listdir(reference_path):
    if fname.endswith('.npy'):
        parts = fname.split('_')
        if len(parts) >= 3:
            key = '_'.join(parts[:3])
            reference_keys.add(key)

# 遍历目标路径文件，找到匹配项并剪切
for fname in os.listdir(target_path):
    if fname.endswith('.npy'):
        parts = fname.split('_')
        if len(parts) >= 3:
            key = '_'.join(parts[:3])
            if key in reference_keys:
                src_file = os.path.join(target_path, fname)
                dst_file = os.path.join(output_path, fname)
                shutil.move(src_file, dst_file)
                print(f'Moved: {fname}')

print('处理完成！')

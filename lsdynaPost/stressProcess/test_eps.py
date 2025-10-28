import numpy as np
import matplotlib.pyplot as plt

# 指定保存的 .npy 文件路径
file_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\eval\original_elementValue_npyFiles\VMStress_Data\Bumper1_dp397_vp400_elementMax_VMStress.npy'

# 加载数据。由于保存的是字典，使用 allow_pickle 参数，并用 .item() 还原为字典对象
# data = np.load(file_path, allow_pickle=True).item()
data = np.load(file_path)
print(data.shape)
print(np.max(data))
data_3d = data.reshape((-1, 15, 500, 500)) #  与LSPP中的bottom视角一致，1~15沿z轴正向的切片
print(np.max(data_3d[100,0,:,:]))
print(np.max(data_3d[100,-1,:,:]))
plt.figure()
plt.imshow(data_3d[100,0,:,:], cmap='magma', aspect='auto')
plt.show()
plt.figure()
plt.imshow(data_3d[100,-1,:,:], cmap='magma', aspect='auto')
plt.show()
eps = np.sum(data_3d, axis=0)
data_3d_front = data_3d[0,:,:]
data_3d_back = data_3d[-1,:,:]
arr = data_3d
print()
print(data_3d.shape)

# for i in range(arr.shape[0]):
#     plt.figure()
#     plt.imshow(arr[i, :, :], cmap='magma', aspect='auto')
#     plt.title(f"Slice at index {i} along axis 2")
#     plt.colorbar()
#     plt.show()

# plt.figure()
# plt.imshow(eps, cmap='magma', aspect='auto')
# plt.title(f"Slice at index eps along axis 2")
# plt.colorbar()
# plt.show()


# 访问数据中的各个字段
# top_coords = data['top_coords']
# bottom_coords = data['bottom_coords']
# t = data['t']
# range_top = data['range_top']
# range_bottom = data['range_bottom']

# 输出部分内容进行验证
# print(top_coords.shape)
# print("上表面轮廓坐标 (前5条):")
# print(top_coords[:5])
# print("\n下表面轮廓坐标 (前5条):")
# print(bottom_coords[:5])
# print("\n指标 t:")
# print(t)
# print("上表面 z 范围:")
# print(range_top)
# print("下表面 z 范围:")
# print(range_bottom)

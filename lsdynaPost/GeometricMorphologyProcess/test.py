import os
import numpy as np
import gc
from lsreader import D3plotReader, DataType as dt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import open3d as o3d
from messagRead import extract_deleted_nodes


'''
备忘：没删导致的 nodeid无间断 20250320
'''

# 定义主路径
main_path = r'\\DESKTOP-SVFNCL2\database\MeshSizeCompare\sph2+RP'
vp_path = r'D:\SIMULATION\test_restart\sph'

d3plot_file = os.path.join(vp_path, 'd3plot')
dr = D3plotReader(d3plot_file)

num_states = dr.get_data(dt.D3P_NUM_STATES)
print(f"Number of states: {num_states}")

# element_VonMisesStress_all = []
temp1 = dr.get_data(dt.D3P_SOLID_HISTORY_VAR, ist=num_states-1, ipt=2, ihv=5, ask_for_numpy_array=True)
print(temp1.shape)  # element数据
print(np.max(temp1),np.min(temp1))
print(temp1)
node = dr.get_data(dt.D3P_NODE_COORDINATES, ist=num_states-1, ipart=2, ask_for_numpy_array=True)
print(node.shape)
element = dr.get_data(dt.D3P_SOLID_ELEMENT_CENTROID, ist=num_states-1, ipart=2, ask_for_numpy_array=True)
print(element.shape)
# element_initial = element_initial[:,2]
# print(element_initial.shape)
# print(np.min(element_initial), np.max(element_initial))
node_end = dr.get_data(dt.D3P_NODE_COORDINATES, ist=num_states - 1, ipart=2, ask_for_numpy_array=True)
# node = node_end
# element_dis[:, 2] = element_end[:, 2] - element_initial[:, 2]
# element_dis = element_dis + element_initial
# 创建空的 PointCloud 对象
pcd_node = o3d.geometry.PointCloud()
pcd_element = o3d.geometry.PointCloud()

# 取出坐标部分（假设至少有 x, y, z 3列）
pcd_node.points = o3d.utility.Vector3dVector(node[:, :3])
print(pcd_node)

#=== 2. 将所有点都设置为灰色 ===#
num_points = len(pcd_node.points)
# 这里的灰度值(0.5, 0.5, 0.5)可以自行调整
# all_grey = np.ones((num_points, 3)) * 0.5
# pcd.colors = o3d.utility.Vector3dVector(all_grey)

#=== 3. 读取需要突出显示的点编号 ===#
# 假设 node_indices.txt 中，每一行是一个点的编号(索引)，从 0 开始
# 如果您的数据中索引是从 1 开始，需要根据实际情况做 -1 处理
node_id_temp, element_id_temp = extract_deleted_nodes(vp_path)
# node_id = int(node_id_temp) - 23984 - 1
# element_id = int(element_id_temp) - 1
node_id = [int(id_str) - 23985 for id_str in node_id_temp]
element_id = [int(id_str) - 23985 for id_str in element_id_temp]

#=== 4. 剔除这些索引对应的点 ===#
# 先构造一个布尔掩码数组，初始全True
mask_node = np.ones(node.shape[0], dtype=bool)
mask_element = np.ones(element.shape[0], dtype=bool)

# 将需要删除的索引位置标记为False
# 如果索引有越界风险，可先做 min/max 校验
mask_node[node_id] = False
mask_element[element_id] = False

# 对points应用掩码，得到剔除后的点集
filtered_points_node = node[mask_node]
filtered_points_element = element[mask_element]

#=== 5. 构建新的点云并可视化 ===#
pcd_filtered_node = o3d.geometry.PointCloud()
pcd_filtered_node.points = o3d.utility.Vector3dVector(filtered_points_node[:, :3])
o3d.visualization.draw_geometries([pcd_filtered_node])
print(pcd_filtered_node)
pcd_filtered_element = o3d.geometry.PointCloud()
pcd_filtered_element.points = o3d.utility.Vector3dVector(filtered_points_element[:, :3])
o3d.visualization.draw_geometries([pcd_filtered_element])
print(pcd_filtered_element)

#=== 5. 可视化 ===#


# element_end = element_end[:,2]
# print(element_end.shape)
# print(np.min(element_end), np.max(element_end))
# element_displacement = element_end - element_initial
# # print(element_displacement.shape)
# print(np.min(element_displacement), np.max(element_displacement))

# fig = plt.figure('123',(10,10))
#
# # create a new subplot on our figure
# # and set projection as 3d
# ax1 = fig.add_subplot(111, projection='3d')
# coord = element_end
# x = coord[:,0]
# y = coord[:,1]
# z = -coord[:,2]
#
# # ax1.ion()
# ax1.scatter( x, y, z, marker = '^')
# plt.axis('equal')
# plt.show()

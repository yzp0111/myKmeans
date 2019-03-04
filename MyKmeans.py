#  encoding=utf-8
'''
用python实现k_means聚类算法
'''
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

np.random.seed(66)
# 计算欧几里得距离
def distance_Euclid(A, B):
	return np.sqrt(np.sum(np.power(A - B, 2)))

# 构建K个聚类中心点（随机质心）
def make_centers(data, k):
	m = np.shape(data)[0]   # 样本个数
	indexs = random.sample(range(m), k)    # 从data中随机取k个样本作为质心
	centers = []    # 存储质心坐标，为了统一数据格式，将值取出至列表在转化为numpy格式
	for i in indexs:
		centers.append(data[i])
	return np.array(centers)

# k_means聚类
def kmeans(data, k):
	data = np.array(data)   # 统一转为numpy.array格式
	m = np.shape(data)[0]   # 样本个数
	# 采用欧式距离作何距离度量方式
	get_distance = distance_Euclid
	# 随机生成k个聚类中心
	centers = make_centers(data, k)
	not_convergence = True    # 是否还未收敛
	while not_convergence:
		cats = []    # 每个样本点各自归属的的类别
		for i in range(m):
			distances = []
			for center in centers:
				distance = get_distance(data[i], center)
				distances.append(distance)
				cat = np.array(distances).argsort()[0]  # 距离最近的中心点
			cats.append(cat)
		# 生成新中心点
		new_centers = []
		for c in range(k):       # c是类别
			mask = np.array([[cat == c, cat == c] for cat in cats])   # 掩码，判断类别是否为c
			data_c = np.ma.masked_array(data, ~mask)
			new_center = data_c.mean(axis=0)
			new_centers.append(new_center)
		new_centers = np.array(new_centers)
		if (new_centers == centers).all():
			break
		else:
			centers = new_centers
	return centers, np.array(cats)

# 预测类别，根据距离中心点的距离，选择距离最近的类别
def predict(data,centers):
	data = np.array(data)   # 统一转为numpy.array格式
	get_distance = distance_Euclid
	cats = []    # 每个样本点各自归属的的类别
	for i in range(np.shape(data)[0]):
		distances = []
		for center in centers:
			distance = get_distance(data[i], center)
			distances.append(distance)
			cat = np.array(distances).argsort()[0]  # 距离最近的中心点
		cats.append(cat)
	return np.array(cats)

# 测试效果
if __name__ == '__main__':
	# 测试数据
	data = pd.read_table('multiple3.txt',header=None,sep=',')
	# 得到训练集中心点及分类结果
	centers, cats = kmeans(data, 4)
	# 用于区分分类边界
	l, r, h = data.iloc[:, 0].min() - 1, data.iloc[:, 0].max() + 1, 0.005
	b, t, v = data.iloc[:, 1].min() - 1, data.iloc[:, 1].max() + 1, 0.005
	grid_x = np.meshgrid(np.arange(l, r, h), np.arange(b, t, v))
	flat_x = np.c_[grid_x[0].ravel(), grid_x[1].ravel()]
	flat_y = predict(flat_x,centers)
	grid_y = flat_y.reshape(grid_x[0].shape)
	# 可视化
	plt.figure('KMeans')
	plt.xlabel('x', fontsize=12)
	plt.ylabel('y', fontsize=12)
	plt.tick_params(labelsize=8)
	plt.tight_layout()
	plt.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='gray')
	plt.scatter(centers[:,0],centers[:,1], marker='+', c='gold', s=1000, linewidth=1)
	plt.scatter(data.iloc[:,0], data.iloc[:,1], c=cats)
	plt.show()

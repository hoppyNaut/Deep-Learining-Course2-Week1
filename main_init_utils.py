"""
1. 初始化参数：
	1.1：使用0来初始化参数。
	1.2：使用随机数来初始化参数。
	1.3：使用抑梯度异常初始化参数（参见视频中的梯度消失和梯度爆炸）。
2. 正则化模型：
	2.1：使用二范数对二分类模型正则化，尝试避免过拟合。
	2.2：使用随机删除节点的方法精简模型，同样是为了尝试避免过拟合。
3. 梯度校验  ：对模型使用梯度校验，检测它是否在梯度下降的过程中出现误差过大的情况
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import gc_utils     # 梯度检验
import init_utils   # 初始化
import reg_utils    # 正则化
import dnn_utils

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot = False)

parameters = dnn_utils.model(train_X, train_Y, initialization = "he")

print("Training Set:")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print("Test Set:")
predictions_test = init_utils.predict(test_X, test_Y, parameters)

print(f'predictions_train = {predictions_train}')
print(f'predictions_test = {predictions_test}')

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda  x:init_utils.predict_dec(parameters,x.T), train_X, train_Y)

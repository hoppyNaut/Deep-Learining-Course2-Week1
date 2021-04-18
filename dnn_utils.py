import numpy as np
import matplotlib.pyplot as plt
import init_utils
import gc_utils
import reg_utils

def initialize_parameters_zeros(layers_dims):
    parameters = {}

    for i in range(1,len(layers_dims)):
        parameters["W" + str(i)] = np.zeros((layers_dims[i], layers_dims[i - 1]))
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))

        assert(parameters["W" + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
        assert(parameters["b" + str(i)].shape == (layers_dims[i], 1))

    return parameters

def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}

    for i in range(1, len(layers_dims)):
        parameters["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * 10
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))

        assert (parameters["W" + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
        assert (parameters["b" + str(i)].shape == (layers_dims[i], 1))

    return parameters

def initialize_parameters_he(layers_dims):
    parameters = {}

    for i in range(1, len(layers_dims)):
        parameters["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * np.sqrt(2 / layers_dims[i])
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))

        assert (parameters["W" + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
        assert (parameters["b" + str(i)].shape == (layers_dims[i], 1))

    return parameters

def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he", is_plot = True):
    """
    实现一个三层的神经网络：ReLU->ReLU->Sigmoid
    :param X:输入数据,维度(2,要训练/测试的数量)
    :param Y:标签,维度(1,对应输入数据的数量)
    :param learning_rate:学习率
    :param num_iterations:迭代次数
    :param print_cost:是否打印成本值
    :param initialization:权重矩阵初始化方法
    :param is_plot:是否绘制梯度下降的曲线图
    :return:
        parameters:更新之后的参数
    """

    parameters = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    # 选择初始化参数类型
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    for i in range(num_iterations):
        # 前向传播
        A3, cache = init_utils.forward_propagation(X, parameters)

        # 计算成本
        cost = init_utils.compute_loss(A3, Y)
        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print(f'第{i}次迭代,成本值为:{cost}')

        # 反向传播
        grads = init_utils.backward_propagation(X, Y, cache)

        # 更新参数
        parameters = init_utils.update_parameters(parameters, grads, learning_rate)

    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters



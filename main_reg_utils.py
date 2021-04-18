import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import gc_utils     # 梯度检验
import init_utils   # 初始化
import reg_utils    # 正则化
import dnn_utils

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    实现成本函数的正则化版本
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = reg_utils.compute_cost(A3, Y)

    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    实现添加L2正则化模型的后向传播
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = 1 / m * np.dot(dZ3, A2.T) + lambd / m * W3
    db3 = 1 / m * np.sum(dZ3, axis = 1, keepdims = True)

    dA2 = np.dot(W3.T, dZ3)
    # relu的求导 np.int64(A2 > 0)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1 / m * np.dot(dZ2, A1.T) + lambd / m * W2
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1 / m * np.dot(dZ1, X.T) + lambd / m * W1
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {"dZ3":dZ3, "dW3":dW3, "db3":db3, "dA2":dA2,
             "dZ2":dZ2, "dW2":dW2, "db2":db2, "dA1":dA1,
             "dZ1":dZ1, "dW1":dW1, "db1":db1}

    return grads


def forward_propagation_with_dropout(X, parameters, keep_prob):
    """
    实现具有随机舍弃节点的dropout
    LINEAR -> RELU + DROPUOUT -> RELU + DROPOUT -> LINEAR -> SIGMOID
    :param X:输入数据集,维度为(2,示例数)
    :param parameters:
        W1 - (20, 2)
        b1 - (20, 1)
        W2 - (3, 20)
        b2 - (3, 1)
        W3 - (1, 3)
        b3 - (1, 1)
    """
    np.random.seed(1)
    L = len(parameters)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = reg_utils.relu(Z1)

    # 初始化矩阵D1
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    # 将D1的值转换为0或1(使用keep_prob作为阈值)
    D1 = D1 < keep_prob
    # 舍弃A1的部分节点
    A1 = A1 * D1
    # 缩放未舍弃的节点
    A1 = A1 / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = reg_utils.relu(Z2)

    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = reg_utils.sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    实现dropout模型的后向传播
    """
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1 / m * np.dot(dZ3, A2.T)
    db3 = 1 / m * np.sum(dZ3, axis = 1, keepdims = True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2
    dA2 = dA2 / keep_prob

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
            "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
            "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return grads





def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, is_plot = True, lambd = 0, keep_prob = 1):
    """
    实现一个三层的神经网络：ReLU->ReLU->Sigmoid
    :param lambd:正则化的超参数
    :param keep_prob:随机删除结点的概率
    :return:
        parameters - 学习后的参数
    """
    parameter = {}
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    # 初始化参数
    parameters = reg_utils.initialize_parameters(layers_dims)

    for l in range(num_iterations):
        # 前向传播
        # 是否随机删除结点
        if keep_prob == 1:
            # 前向传播
            a3, cache = reg_utils.forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        else:
            print("keep_prob参数错误!程序退出")
            exit()

        # 计算成本
        # 是否使用正则化
        if lambd == 0:
            cost = reg_utils.compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        if l % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print(f'第{l}次迭代,成本值为:{cost}')

        # 反向传播
        # 可以同时使用L2正则化和随机删除节点,但本次实验不同时使用
        assert(lambd == 0 or keep_prob == 1)

        if lambd == 0 and keep_prob == 1:
            # 不使用L2正则化和随机删除节点
            grads = reg_utils.backward_propagation(X, Y, cache)
        elif lambd != 0:
            # 使用L2正则化,不使用dropout
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            # 使用dropout,不使用L2正则化
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # 更新参数
        parameters = reg_utils.update_parameters(parameters, grads, learning_rate)

    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=False)

parameters = model(train_X, train_Y, keep_prob= 0.86,is_plot=True)
print("使用dropout，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("使用dropout，测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
import numpy as np
import random
import math


class funtion():
    def __init__(self):
        print("starting DBO")


def Parameters(F):
    if F == 'F1':
        # ParaValue=[-100,100,30] [-100,100]代表初始范围，30代表dim维度
        fobj = F1
        lb = -100
        ub = 100
        dim = 30
    elif F == 'F2':
        fobj = F2
        lb = -10
        ub = 10
        dim = 30
    elif F == 'F3':
        fobj = F3
        lb = -100
        ub = 100
        dim = 30
    elif F == 'F4':
        fobj = F4
        lb = -100
        ub = 100
        dim = 30
    elif F == 'F5':
        fobj = F5
        lb = -30
        ub = 30
        dim = 30
    elif F == 'F6':
        fobj = F6
        lb = -100
        ub = 100
        dim = 30
    elif F == 'F7':
        fobj = F7
        lb = -1.28
        ub = 1.28
        dim = 30
    elif F == 'F8':
        fobj = F8
        lb = -500
        ub = 500
        dim = 30
    elif F == 'F9':
        fobj = F9
        lb = -5.12
        ub = 5.12
        dim = 30

    elif F == 'F10':
        fobj = F10
        lb = -32
        ub = 32
        dim = 30

    elif F == 'F11':
        fobj = F11
        lb = -600
        ub = 600
        dim = 30

    elif F == 'F12':
        fobj = F12
        lb = -50
        ub = 50
        dim = 30

    elif F == 'F13':
        fobj = F13
        lb = -50
        ub = 50
        dim = 30

    elif F == 'F14':
        fobj = F14
        lb = -65.536
        ub = 65.536
        dim = 2

    elif F == 'F15':
        fobj = F15
        lb = -5
        ub = 5
        dim = 4

    elif F == 'F16':
        fobj = F16
        lb = -5
        ub = 5
        dim = 2

    elif F == 'F17':
        fobj = F17
        lb = [-5, 0]
        ub = [10, 15]
        dim = 2

    elif F == 'F18':
        fobj = F18
        lb = -2
        ub = 2
        dim = 2

    elif F == 'F19':
        fobj = F19
        lb = 0
        ub = 1
        dim = 3

    elif F == 'F20':
        fobj = F20
        lb = 0
        ub = 1
        dim = 6

    elif F == 'F21':
        fobj = F21
        lb = 0
        ub = 10
        dim = 4

    elif F == 'F22':
        fobj = F22
        lb = 0
        ub = 10
        dim = 4

    elif F == 'F23':
        fobj = F23
        lb = 0
        ub = 10
        dim = 4;
    return fobj, lb, ub, dim


# F1

def F1(x):
    o = np.sum(np.square(x))
    return o


# F2
def F2(x):
    o = np.sum(np.abs(x)) + np.prod(np.abs(x))
    return o


# F3
def F3(x):
    dim = len(x)
    o = 0
    for i in range(dim):
        o = o + np.square(np.sum(x[0:i]))
    return o


# F4
def F4(x):
    o = np.max(np.abs(x))
    return o


# F5
def F5(x):
    dim = len(x)
    o = np.sum(100 * np.square(x[1:dim] - np.square(x[0:dim - 1]))) + np.sum(np.square(x[0:dim - 1] - 1))
    return o


# F6
def F6(x):
    o = np.sum(np.square(np.abs(x + 0.5)))
    return o


# F7
def F7(x):
    dim = len(x)
    num1 = [num for num in range(1, dim + 1)]
    o = np.sum(num1 * np.power(x, 4)) + np.random.rand(1)
    return 0


# F8
def F8(x):
    o = np.sum(0 - x * np.sin(np.sqrt(np.abs(x))))
    return 0


# F9
def F9(x):
    dim = len(x)
    o = np.sum(np.square(x) - 10 * np.cos(2 * math.pi * x)) + 10 * dim
    return 0


# F10
def F10(x):
    dim = len(x)
    o = 0 - 20 * np.exp(0 - 0.2 * np.sqrt(np.sum(np.square(x)) / dim)) - np.exp(
        np.sum(np.cos(2 * math.pi * x)) / dim) + 20 + np.exp(1)
    return 0


# F11
def F11(X):
    dim=len(X)
    i = np.arange(1,dim+1)
    O=np.sum(X**2)/4000-np.prod(np.cos(X/np.sqrt(i)))+1
    return O


# F12
def F12(x):
    dim = len(x)
    o = (math.pi / dim) * (10 * np.square(np.sin(math.pi * (1 + (x[1] + 1) / 4))) + \
                           np.sum((((x[0:dim - 2] + 1) / 4) ** 2) * \
                                  (1 + 10. * ((np.sin(math.pi * (1 + (x[1:dim - 1] + 1) / 4)))) ** 2)) + (
                                       (x[dim - 1] + 1) / 4) ** 2) + np.sum(Ufun(x, 10, 100, 4))
    return o


# F13
def F13(x):
    dim = len(x)
    o = 0.1 * (np.square(np.sin(3 * math.pi * x[1])) + np.sum(
        np.square(x[0:dim - 2] - 1) * (1 + np.square(np.sin(3 * math.pi * x[1:dim - 1])))) + \
               np.square(x[dim - 1] - 1) * (1 + np.square(np.sin(2 * math.pi * x[dim - 1])))) + np.sum(
        Ufun(x, 5, 100, 4))

    return o


def Ufun(x, a, k, m):
    o = k * np.power(x - a, m) * (x > a) + k * (np.power(-x - a, m)) * (x < (-a))
    return o


# F14
def F14(x):
    aS = [[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32], \
          [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]]
    ss = x.T
    bS = np.zeros
    bS = aS[0, :]
    num1 = [num for num in range(1, 26)]
    for j in range(25):
        bS[j] = np.sum(np.power(ss - aS[:, j], 6))
    o = 1 / (1 / 500 + np.sum(1 / (num1 + bS)))
    return o


# F15
def F15(x):
    aK = np.array[0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
    bK = np.array[0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    bK = 1 / bK
    o = np.sum(np.square(aK - (x[1] * (np.square(bK) + x[2] * bK)) / (np.square(bK) + x[3] * bK + x[4])))
    return o


# F16
def F16(x):
    o = 4 * np.square(x[1]) - 2.1 * np.power(x[1], 4) + np.power(x[1], 6) / 3 + \
        x[1] * x[2] - 4 * np.square(x[2]) + 4 * np.power(x[2], 4)
    return o


# F17
def F17(x):
    o = np.square(x[2] - np.square(x[1]) * 5.1 / (4 * np.square(math.pi)) + 5 / math.pi * x[1] - 6) + \
        10 * (1 - 1 / (8 * math.pi)) * np.cos(x[1]) + 10
    return o


# F18
def F18(x):
    o = (1 + np.square(x[1] + x[2] + 1) * (
                19 - 14 * x[1] + 3 * np.square(x[1]) - 14 * x[2] + 6 * x[1] * x[2] + 3 * np.square(x[2]))) * \
        (30 + np.square(2 * x[1] - 3 * x[2]) * (
                    18 - 32 * x[1] + 12 * np.square(x[1]) + 48 * x[2] - 36 * x[1] * x[2] + 27 * np.square(x[2])))
    return o


# F19
def F19(x):
    aH = np.array[[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
    cH = np.array[1, 1.2, 3, 3.2]
    pH = np.array[[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]]
    o = 0;
    for i in range(4):
        o = o - cH[i] * np.exp(0 - np.sum(aH[i, :] * np.square(x - pH[i, :])))
    return o


# F20
def F20(x):
    aH = np.array[
        [10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
    cH = np.array[1, 1.2, 3, 3.2]
    pH = np.array[
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991], [0.2348,
                                                                                                             0.1415,
                                                                                                             0.3522,
                                                                                                             0.2883,
                                                                                                             0.3047,
                                                                                                             0.6650], [
            0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]
    o = 0
    for i in range(4):
        o = o - cH(i) * np.exp(0 - (np.sum(aH[i, :] * (np.square(x - pH[i, :])))))
    return o


# F21
def F21(x):
    aSH = np.array[
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8,
                                                                                                           1], [6, 2, 6,
                                                                                                                2], [7,
                                                                                                                     3.6,
                                                                                                                     7,
                                                                                                                     3.6]]
    cSH = np.array[0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    o = 0
    for i in range(5):
        o = o - 1 / (np.sum((x - aSH[i, :]) * (x - aSH[i, :])) + cSH(i))
    return o


# F22
def F22(x):
    aSH = np.array[
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8,
                                                                                                           1], [6, 2, 6,
                                                                                                                2], [7,
                                                                                                                     3.6,
                                                                                                                     7,
                                                                                                                     3.6]]
    cSH = np.array[0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    o = 0
    for i in range(7):
        o = o - 1 / (np.sum((x - aSH[i, :]) * (x - aSH[i, :])) + cSH(i))
    return o


# F23
def F23(x):
    aSH = np.array[
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8,
                                                                                                           1], [6, 2, 6,
                                                                                                                2], [7,
                                                                                                                     3.6,
                                                                                                                     7,
                                                                                                                     3.6]]
    cSH = np.array[0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    o = 0
    for i in range(10):
        o = o - 1 / (np.sum((x - aSH[i, :]) * (x - aSH[i, :])) + cSH[i])
    return o


def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:
            temp[i] = Lb[0, i]
        elif temp[i] > Ub[0, i]:
            temp[i] = Ub[0, i]

    return temp
def Boundss(ss, LLb, UUb):
    temp = ss
    for i in range(len(ss)):
        if temp[i] < LLb[0, i]:
            temp[i] = LLb[0, i]
        elif temp[i] > UUb[0, i]:
            temp[i] = UUb[0, i]
    return temp


def swapfun(ss):
    temp = ss
    o = np.zeros((1, len(temp)))
    for i in range(len(ss)):
        o[0, i] = temp[i]
    return o


def DBO(pop, M, c, d, dim, fun):
    """
    :param fun: 适应度函数
    :param pop: 种群数量
    :param M: 迭代次数
    :param c: 迭代范围下界
    :param d: 迭代范围上界
    :param dim: 优化参数的个数
    :return: 适应度值最小的值 对应得位置
    """
    P_percent1 = 0.2 # 初始化滚球蜣螂比例
    P_percent2 = 0.4 # 初始化滚球+产卵+蜣螂比例
    P_percent3 = 0.65 # 初始化滚球+产卵+觅食蜣螂比例
    pNum1 = round(pop * P_percent1)
    pNum2 = round(pop * P_percent2)
    pNum3 = round(pop * P_percent3)

    lb = c*np.ones((1, dim))  # 下界
    ub = d*np.ones((1, dim))  # 上界

    # 使用Bernoulli混沌序列初始化种群
    X = np.random.rand(pop, dim)
    lambda_ = 0.4  # 映射参数
    X[X <= (1 - lambda_)] /= (1 - lambda_)
    X[X > (1 - lambda_)] = (X[X > (1 - lambda_)] - 1 + lambda_) / lambda_

    fit = np.zeros((pop, 1))  # 初始化适应度值

    for i in range(pop):  # 对种群中的每个个体进行操作
        X[i,:] = lb+(ub-lb)*X[i,:]  # 随机生成初始种群
        fit[i:,0] = fun(X[i,:])  # 计算适应度值

    pFit = fit  # 个体最优适应度值
    pX = X  # 个体最优位置
    XX = pX  # 个体最优位置的副本
    fMin = np.min(fit[:, 0])  # 全局最优适应度值
    bestI = np.argmin(fit[:, 0])  # 全局最优适应度值的位置
    bestX = X[bestI, :]  # 全局最优位置
    Convergence_curve = np.zeros((1, M))  # 收敛曲线

    for t in range(M):  # 开始迭代
        fmax = np.max(pFit[:, 0])  # 最大适应度值
        B = np.argmax(pFit[:, 0])  # 最大适应度值的位置
        worse = X[B, :]  # 最差个体
        w_max = 0.9
        w_min = 0.782
        r1 = (w_max - w_min) / 2 * np.cos(np.pi * t / M) + (w_max + w_min) / 2
        r2 = 2 * np.pi * np.random.rand(1) # 生成[0,2pi]上的随机数
        r3 = np.random.rand() * 4 - 2 # 生成[-2,2]上的随机数
        w = w_max - (w_max - w_min) * (t / M)
        r4 = np.random.rand()  # 随机数
        for i in range(pNum1):  # 对种群中的每个个体进行操作
            if r4 < 0.5:  # 如果随机数小于0.9
                b = 0.3
                k = 0.1
                a = np.random.rand()  # 生成新的随机数
                if a > 0.1:  # 如果随机数大于0.1
                    a = 1  # 设定a为1
                else:
                    a = -1  # 否则设定a为-1
                X[i, :] = pX[i, :] + b * np.abs(pX[i, :] - worse) + a * k * (XX[i, :])  # 滚球行为，利用公式1更新位置
            else:  # 如果随机数大于等于0.5
                '''标准的跳舞行为'''
                # aaa = np.random.randint(180, size=1)  # 生成一个随机整数
                # if aaa == 0 or aaa == 90 or aaa == 180:  # 如果随机数是0，90或180
                #     X[i, :] = pX[i, :]  # 位置保持不变
                # theta = aaa * math.pi / 180  # 将随机数转换为弧度
                # X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])  # 跳舞行为，利用公式2更新位置
                '''用改进正弦函数MSA改善跳舞行为'''
                for j in range(dim):
                    X[i, j] = w * X[i, j] + (r1 * np.sin(r2) * (r3 * bestX[j] - X[i, j]))
            X[i, :] = Bounds(X[i, :], lb, ub)  # 确保新位置在边界内
            fit[i, 0] = fun(X[i, :])  # 计算新位置的适应度值
        bestII = np.argmin(fit[:, 0])  # 找到最优适应度值的位置
        bestXX = X[bestII, :]  # 找到最优位置

        # 产卵行为
        R = 1 - t / M  # 计算R
        Xnew1 = bestXX * (1 - R)  # 计算Xnew1
        Xnew2 = bestXX * (1 + R)  # 计算Xnew2
        Xnew1 = Bounds(Xnew1, lb, ub)  # 模拟雌性蜣螂产卵的区域，确保Xnew1在边界内，利用公式3
        Xnew2 = Bounds(Xnew2, lb, ub)  # 确保Xnew2在边界内
        Xnew11 = bestX * (1 - R)  # 计算Xnew11 # 模拟觅食行为，觅食区域的下限，利用公式5
        Xnew22 = bestX * (1 + R)  # 计算Xnew22 # 觅食行为的上限
        Xnew11 = Bounds(Xnew11, lb, ub)  # 确保Xnew11在边界内
        Xnew22 = Bounds(Xnew22, lb, ub)  # 确保Xnew22在边界内
        xLB = swapfun(Xnew1)  # 调用swapfun函数处理Xnew1
        xUB = swapfun(Xnew2)  # 调用swapfun函数处理Xnew2

        for i in range(pNum1 + 1, pNum2):  # 对种群中的每个个体进行操作，利用公式4
            X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)  # 更新位置
            X[i, :] = Bounds(X[i, :], xLB, xUB)  # 确保新位置在边界内
            fit[i, 0] = fun(X[i, :])  # 计算新位置的适应度值
        for i in range(pNum2+1, pNum3):  # 对种群中的每个个体进行操作，利用公式6
            X[i, :] = pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))  # 更新位置
            X[i, :] = Bounds(X[i, :], lb, ub)  # 确保新位置在边界内
            fit[i, 0] = fun(X[i, :])  # 计算新位置的适应度值
        for j in range(pNum3+1, pop):  # 对种群中的每个个体进行操作，利用公式7
            X[j, :] = bestX + np.random.randn(1, dim) * (
                        np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2  # 更新位置
            X[j, :] = Bounds(X[j, :], lb, ub)  # 确保新位置在边界内
            fit[j, 0] = fun(X[j, :])  # 计算新位置的适应度值

        # 更新个体的最佳适应度值和全局最佳适应度值
        XX = pX  # 更新个体最优位置的副本
        for i in range(pop):  # 对种群中的每个个体进行操作
            if fit[i, 0] < pFit[i, 0]:  # 如果新位置的适应度值小于个体最优适应度值
                pFit[i, 0] = fit[i, 0]  # 更新个体最优适应度值
                pX[i, :] = X[i, :]  # 更新个体最优位置

            # 自适应高斯-柯西扰动变异
                w1 = t/M
                w2 = 1 - w1
                X[i, :] = pX[i, :] * (
                        1 + w1 * np.random.randn() + w2 * np.tan((np.random.rand() - 1 / 2) * np.pi))  # 高斯-柯西扰动变异
                X[i, :] = Bounds(X[i, :], lb, ub)  # 控制蜣螂在边界
                fit[i,0] = fun(X[i, :])

            if fit[i, 0] < pFit[i, 0]:  #重新评估
                pFit[i, 0] = fit[i, 0]
                pX[i, :] = X[i, :]

            if pFit[i, 0] < fMin:  # 如果个体最优适应度值小于全局最优适应度值
                fMin = pFit[i, 0]  # 更新全局最优适应度值
                bestX = pX[i, :]  # 更新全局最优位置

        Convergence_curve[0, t] = fMin  # 更新收敛曲线
        print('MSADBO: 迭代', t, '次后，最佳适应度是', Convergence_curve[0,t])

    return fMin, bestX, Convergence_curve  # 返回全局最优适应度值，全局最优位置和收敛曲线


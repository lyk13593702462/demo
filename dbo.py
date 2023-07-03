#载入所需的包
import numpy as np
import random
import copy
import math


'''初始化蜣螂种群'''
def chaotic_map(z, beta): #使用混沌映射初始化种群，这里采用Bernoulli映射
    if 0 <= z <= 1 - beta:
        return z / (1 - beta)
    elif 1 - beta <= z <= 1:
        return (z - (1 - beta)) / beta
    
def initial(pop, dim, ub, lb, beta=0.518):
    X = np.zeros([pop, dim])
    z = 0.326 # 初始化z为一个在[0, 1]范围内的随机数，或者自定义
    
    for i in range(pop):
        for j in range(dim):
            z = chaotic_map(z, beta)
            # 将z的值缩放到 [lb, ub] 的范围内
            X[i, j] = z * (ub[0] - lb[0]) + lb[0]
            
    return X, lb, ub
    
'''定义适应度函数'''
def fun(X):
    O = 0
    for i in X:
        O += i ** 2
    return O

'''计算适应度函数'''
def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

'''适应度排序'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew

'''利用改进正弦算法改善蜣螂滚球行为与舞蹈行为的位置更新'''
def inertia_weight(omega_max, omega_min, t, t_max):
    return omega_max - (omega_max - omega_min) * (t / t_max)

def nonlinear_decreasing(omega_max, omega_min, t, t_max):
    return ((omega_max - omega_min) / 2) * math.cos((math.pi * t) / t_max) + (omega_max + omega_min) / 2

def BRUpdate(X, XLast, pNum, GworstPosition, best_positions, omega_max, omega_min, t, t_max, ST, lb, ub):
    X_new = copy.copy(X)
    r2 = np.random.rand()
    dim = X.shape[1]
    b = 0.3
    
    omega_t = inertia_weight(omega_max, omega_min, t, t_max)
    r1 = nonlinear_decreasing(omega_max, omega_min, t, t_max)
    
    for i in range(pNum):
        delta = np.random.rand(1)
        if delta < ST:
            a = np.random.rand()
            if a > 0.1:
                a = 1
            else:
                a = -1
            X_new[i, :] = X[i, :] + b * np.abs(X[i, :] - GworstPosition[0,:]) + a * 0.1 * (XLast[i, :])
        else:
            r2 = np.random.uniform(0, 2 * math.pi)
            r3 = np.random.uniform(-2, 2)
            X_new[i, :] = omega_t * X[i, :] + r1 * math.sin(r2) * (r3 * best_positions[i, :] - X[i, :])
    
    # 防止产生蜣螂位置超出搜索空间
    for i in range(pNum):
        for j in range(dim):
            X_new[i, j] = np.clip(X_new[i, j], lb[j], ub[j])
            
    return X_new

'''麻雀加入者勘探更新'''
'''蜣螂繁殖行为'''
def SPUpdate(X, pNum, t, Max_iter, fitness):
    X_new = copy.copy(X)
    dim = X.shape[1]
    R = 1 - t / Max_iter
    fMin = np.min(fitness)  # 找到X中最小的适应度
    bestIndex = np.argmin(fitness)  # 找到X中最小适应度的索引
    bestX = X[bestIndex, :]  # 找到X中具有最有适应度的蜣螂位置
    lbStar = bestX * (1 - R)
    ubStar = bestX * (1 + R)
    for j in range(dim):
        lbStar[j] = np.clip(lbStar[j], lb[j], ub[j])  # Equation(3)
        ubStar[j] = np.clip(ubStar[j], lb[j], ub[j])  # Equation(3)
    # XLb = swapfun(lbStar)
    # XUb = swapfun(ubStar)
    for i in range(pNum + 1, 12):  # Equation(4)
        X_new[i, :] = bestX + (np.random.rand(1, dim)) * (
                X[i, :] - lbStar + (np.random.rand(1, dim)) * (X[i, :] - ubStar))
        for j in range(dim):
            X_new[i, j] = np.clip(X_new[i, j], lbStar[j], ubStar[j])
    return X_new

'''蜣螂觅食行为'''
def FAUpdate(X, t, Max_iter, GbestPosition):
    X_new = copy.copy(X)
    dim = X.shape[1]
    R = 1 - t / Max_iter
    lbb = GbestPosition[0, :] * (1 - R)
    ubb = GbestPosition[0, :] * (1 + R)
    for j in range(dim):
        lbb[j] = np.clip(lbb[j], lb[j], ub[j])  # Equation(5)
        ubb[j] = np.clip(ubb[j], lb[j], ub[j])  # Equation(5)
    for i in range(13, 19):  # Equation(6)
        X_new[i, :] = X[i, :] + (np.random.rand(1, dim)) * (X[i, :] - lbb) + (np.random.rand(1, dim)) * (X[i, :] - ubb)
        for j in range(dim):
            X_new[i, j] = np.clip(X_new[i, j], lbb[j], ubb[j])
    return X_new

'''蜣螂偷窃行为'''
def THUpdate(X, t, Max_iter, GbestPosition, fitness):
    X_new = copy.copy(X)
    dim = X.shape[1]
    fMin = np.min(fitness)  # 找到X中最小的适应度
    bestIndex = np.argmin(fitness)  # 找到X中最小适应度的索引
    bestX = X[bestIndex, :]  # 找到X中具有最有适应度的蜣螂位置
    for i in range(20, pop):  # Equation(7)
        X_new[i, :] = GbestPosition[0, :] + np.random.randn(1, dim) * (
                np.abs(X[i, :] - GbestPosition[0, :]) + np.abs(X[i, :] - bestX)) / 2
        for j in range(dim):
            X_new[i, j] = np.clip(X_new[i, j], lb[j], ub[j])
    return X_new

'''高斯-柯西变异扰动函数'''
def gaussian_cauchy_mutation(X_best, sigma, t, T_max):
    # 计算权重系数
    mu1 = t / T_max
    mu2 = 1 - mu1
    
    # 执行高斯-柯西混合变异
    return X_best * (1 + mu1 * norm.rvs(scale=sigma) + mu2 * cauchy.rvs(scale=sigma))

'''蜣螂优化算法'''
def DBO(pop, dim, lb, ub, Max_iter, fun, sigma=0.1):
    """
        :param fun: 适应度函数
        :param pop: 种群数量
        :param Max_iter: 迭代次数
        :param lb: 迭代范围下界
        :param ub: 迭代范围上界
        :param dim: 优化参数的个数
        :return: GbestScore、GbestPosition、Curve : 适应度值最小的值、对应的位置、迭代过程中的历代最优位置
    """
    P_percent = 0.2
    pNum = round(pop * P_percent)
    X, lb, ub = initial(pop, dim, ub, lb)
    fitness = CaculateFitness(X, fun)
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex)
    XLast = X  # X(t-1)
    GbestScore = copy.copy(fitness[0])
    GbestPosition = np.zeros([1, dim])
    GbestPosition[0, :] = copy.copy(X[0, :])

    GworstScore = copy.copy(fitness[-1])
    GworstPosition = np.zeros([1, dim])
    GworstPosition[0, :] = copy.copy(X[-1, :])

    Curve = np.zeros([Max_iter, 1])

    for t in range(Max_iter):
        BestF = fitness[0]
        X = BRUpdate(X, XLast, pNum, GworstPosition)  # 滚球和舞蹈行为
        fitness = CaculateFitness(X, fun)  # 重新计算并排序
        X = SPUpdate(X, pNum, t, Max_iter, fitness)
        X = FAUpdate(X, t, Max_iter, GbestPosition)
        fitness = CaculateFitness(X, fun)  # 重新计算并排序
        X = THUpdate(X, t, Max_iter, GbestPosition, fitness)
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        XLast = X
        
        # 添加自适应高斯-柯西变异策略
        mutated_solution = gaussian_cauchy_mutation(GbestPosition, sigma, t, Max_iter)
        mutated_fitness = fun(mutated_solution)
        
        # 使用贪婪规则判断是否接受新解
        if mutated_fitness < GbestScore:
            GbestScore = mutated_fitness
            GbestPosition = mutated_solution
            
        if (fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPosition[0, :] = copy.copy(X[0, :])
        Curve[t] = GbestScore

        if (t) % 50 == 0:
            print("第%d代搜索的结果为:%f" % (t, GbestScore[0]))
    return GbestScore, GbestPosition, Curve
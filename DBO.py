import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

'''基于正弦算法引导的蜣螂优化算法'''
class MSADBOAlgorithm:
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

    # 定义SVR的适应度函数
    def fitness_function(parameters,X, y):
        C, gamma = parameters
        svr = SVR(C=C, gamma=gamma, kernel='rbf')
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(svr, X, y, cv=cv, scoring='neg_mean_absolute_error')  # 使用负MAE作为评分，因为优化算法通常寻找最大值
        return np.mean(scores)  # 返回平均得分

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
        P_percent1 = 0.2  # 初始化滚球蜣螂比例
        P_percent2 = 0.4  # 初始化滚球+产卵+蜣螂比例
        P_percent3 = 0.65  # 初始化滚球+产卵+觅食蜣螂比例
        pNum1 = round(pop * P_percent1)
        pNum2 = round(pop * P_percent2)
        pNum3 = round(pop * P_percent3)

        lb = c * np.ones((1, dim))  # 下界
        ub = d * np.ones((1, dim))  # 上界

        # 使用Bernoulli混沌序列初始化种群
        X = np.random.rand(pop, dim)
        lambda_ = 0.4  # 映射参数
        X[X <= (1 - lambda_)] /= (1 - lambda_)
        X[X > (1 - lambda_)] = (X[X > (1 - lambda_)] - 1 + lambda_) / lambda_

        fit = np.zeros((pop, 1))  # 初始化适应度值

        for i in range(pop):  # 对种群中的每个个体进行操作
            X[i, :] = lb + (ub - lb) * X[i, :]  # 随机生成初始种群
            fit[i:, 0] = fun(X[i, :])  # 计算适应度值

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
            r2 = 2 * np.pi * np.random.rand(1)  # 生成[0,2pi]上的随机数
            r3 = np.random.rand() * 4 - 2  # 生成[-2,2]上的随机数
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
                X[i, :] = MSADBOAlgorithm.Bounds(X[i, :], lb, ub)  # 确保新位置在边界内
                fit[i, 0] = fun(X[i, :])  # 计算新位置的适应度值
            bestII = np.argmin(fit[:, 0])  # 找到最优适应度值的位置
            bestXX = X[bestII, :]  # 找到最优位置

            # 产卵行为
            R = 1 - t / M  # 计算R
            Xnew1 = bestXX * (1 - R)  # 计算Xnew1
            Xnew2 = bestXX * (1 + R)  # 计算Xnew2
            Xnew1 = MSADBOAlgorithm.Bounds(Xnew1, lb, ub)  # 模拟雌性蜣螂产卵的区域，确保Xnew1在边界内，利用公式3
            Xnew2 = MSADBOAlgorithm.Bounds(Xnew2, lb, ub)  # 确保Xnew2在边界内
            Xnew11 = bestX * (1 - R)  # 计算Xnew11 # 模拟觅食行为，觅食区域的下限，利用公式5
            Xnew22 = bestX * (1 + R)  # 计算Xnew22 # 觅食行为的上限
            Xnew11 = MSADBOAlgorithm.Bounds(Xnew11, lb, ub)  # 确保Xnew11在边界内
            Xnew22 = MSADBOAlgorithm.Bounds(Xnew22, lb, ub)  # 确保Xnew22在边界内
            xLB = MSADBOAlgorithm.swapfun(Xnew1)  # 调用swapfun函数处理Xnew1
            xUB = MSADBOAlgorithm.swapfun(Xnew2)  # 调用swapfun函数处理Xnew2

            for i in range(pNum1 + 1, pNum2):  # 对种群中的每个个体进行操作，利用公式4
                X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)  # 更新位置
                X[i, :] = MSADBOAlgorithm.Bounds(X[i, :], xLB, xUB)  # 确保新位置在边界内
                fit[i, 0] = fun(X[i, :])  # 计算新位置的适应度值
            for i in range(pNum2 + 1, pNum3):  # 对种群中的每个个体进行操作，利用公式6
                X[i, :] = pX[i, :] + ((np.random.randn(1)) * (pX[i, :] - Xnew11) + (
                        (np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))  # 更新位置
                X[i, :] = MSADBOAlgorithm.Bounds(X[i, :], lb, ub)  # 确保新位置在边界内
                fit[i, 0] = fun(X[i, :])  # 计算新位置的适应度值
            for j in range(pNum3 + 1, pop):  # 对种群中的每个个体进行操作，利用公式7
                X[j, :] = bestX + np.random.randn(1, dim) * (
                        np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2  # 更新位置
                X[j, :] = MSADBOAlgorithm.Bounds(X[j, :], lb, ub)  # 确保新位置在边界内
                fit[j, 0] = fun(X[j, :])  # 计算新位置的适应度值

            # 更新个体的最佳适应度值和全局最佳适应度值
            XX = pX  # 更新个体最优位置的副本
            for i in range(pop):  # 对种群中的每个个体进行操作
                if fit[i, 0] < pFit[i, 0]:  # 如果新位置的适应度值小于个体最优适应度值
                    pFit[i, 0] = fit[i, 0]  # 更新个体最优适应度值
                    pX[i, :] = X[i, :]  # 更新个体最优位置

                    # 自适应高斯-柯西扰动变异
                    w1 = t / M
                    w2 = 1 - w1
                    X[i, :] = pX[i, :] * (
                            1 + w1 * np.random.randn() + w2 * np.tan((np.random.rand() - 1 / 2) * np.pi))  # 高斯-柯西扰动变异
                    X[i, :] = MSADBOAlgorithm.Bounds(X[i, :], lb, ub)  # 控制蜣螂在边界
                    fit[i, 0] = fun(X[i, :])

                if fit[i, 0] < pFit[i, 0]:  # 重新评估
                    pFit[i, 0] = fit[i, 0]
                    pX[i, :] = X[i, :]

                if pFit[i, 0] < fMin:  # 如果个体最优适应度值小于全局最优适应度值
                    fMin = pFit[i, 0]  # 更新全局最优适应度值
                    bestX = pX[i, :]  # 更新全局最优位置

            Convergence_curve[0, t] = fMin  # 更新收敛曲线
            print('MSADBO: 迭代', t, '次后，最佳适应度是', Convergence_curve[0, t])

        return fMin, bestX, Convergence_curve  # 返回全局最优适应度值，全局最优位置和收敛曲线
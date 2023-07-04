import numpy as np

# 边界处理函数
def Bounds(s, Lb, Ub):
    temp = s.copy()
    temp[temp < Lb] = Lb[temp < Lb]
    temp[temp > Ub] = Ub[temp > Ub]
    return temp

# 蜣螂算法主函数
def MSADBO(N, Max_iter, lb, ub, dim, fobj):
    P_percent1 = 0.2
    P_percent2 = 0.4
    P_percent3 = 0.65
    pNum1 = round(N * P_percent1)
    pNum2 = round(N * P_percent2)
    pNum3 = round(N * P_percent3)

    # 初始化种群
    Z = np.random.rand(N, dim)
    lambda_ = 0.518
    Z = np.where((Z <= (1 - lambda_)) & (Z > 0), Z / (1 - lambda_), (Z - 1 + lambda_) / lambda_)
    x = lb + (ub - lb) * Z
    fit = np.apply_along_axis(fobj, 1, x)

    pFit = fit.copy()
    pX = x.copy()
    XX = pX.copy()
    fMin = np.min(fit)
    bestX = x[np.argmin(fit), :]

    # 开始迭代
    for t in range(Max_iter):
        Wmax = 0.9
        Wmin = 0.782
        r1 = (Wmax - Wmin) / 2 * np.cos(np.pi * t / Max_iter) + (Wmax + Wmin) / 2
        w = Wmax - (Wmax - Wmin) * (t / Max_iter)
        worse = x[np.argmax(fit), :]

        for i in range(pNum1):
            if np.random.rand(1) < 0.5:
                b = 0.3
                k = 0.1
                r4 = np.random.rand()
                miu = 0.1
                a = 1 if r4 > miu else -1
                x[i, :] = pX[i, :] + b * np.abs(pX[i, :] - worse) + a * k * XX[i, :]
            else:
                r2 = 2 * np.pi * np.random.rand()
                r3 = 2 * np.random.rand()
                x[i, :] = w * x[i, :] + (r1 * np.sin(r2) * (r3 * bestX - x[i, :]))

            x[i, :] = Bounds(x[i, :], lb, ub)
            fit[i] = fobj(x[i, :])

        bestII = np.argmin(fit)
        bestXX = x[bestII, :]
        R = 1 - t / Max_iter
        Xnew1 = bestXX * (1 - R)
        Xnew2 = bestXX * (1 + R)
        Xnew1 = Bounds(Xnew1, lb, ub)
        Xnew2 = Bounds(Xnew2, lb, ub)
        Xnew11 = bestX * (1 - R)
        Xnew22 = bestX * (1 + R)
        lbnew2 = Bounds(Xnew11, lb, ub)
        ubnew2 = Bounds(Xnew22, lb, ub)

        for i in range(pNum1, pNum2):
            x[i, :] = bestXX + (np.random.rand(1, dim) * (pX[i, :] - Xnew1) + np.random.rand(1, dim) * (pX[i, :] - Xnew2))
            x[i, :] = Bounds(x[i, :], Xnew1, Xnew2)
            fit[i] = fobj(x[i, :])

        for i in range(pNum2, pNum3):
            x[i, :] = pX[i, :] + (np.random.randn(1) * (pX[i, :] - lbnew2) + (np.random.rand(1, dim) * (pX[i, :] - ubnew2)))
            x[i, :] = Bounds(x[i, :], lb, ub)
            fit[i] = fobj(x[i, :])

        for j in range(pNum3, N):
            s = 1
            x[j, :] = bestX + s * np.random.randn(1, dim) * ((np.abs(pX[j, :] - bestXX)) + (np.abs(pX[j, :] - bestX))) / 2
            x[j, :] = Bounds(x[j, :], lb, ub)
            fit[j] = fobj(x[j, :])

        XX = pX.copy()
        for i in range(N):
            w1 = t / Max_iter
            w2 = 1 - t / Max_iter
            x[i, :] = pX[i, :] * (1 + w1 * np.random.randn() + w2 * np.tan((np.random.rand() - 1 / 2) * np.pi))
            x[i, :] = Bounds(x[i, :], lb, ub)
            fit[i] = fobj(x[i, :])

            if fit[i] < pFit[i]:
                pFit[i] = fit[i]
                pX[i, :] = x[i, :]

            if pFit[i] < fMin:
                fMin = pFit[i]
                bestX = pX[i, :]

    return fMin, bestX
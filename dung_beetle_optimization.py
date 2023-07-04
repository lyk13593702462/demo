import numpy as np


def bounds(x, lb, ub):
    return np.clip(x, lb, ub)


def MSADBO(N, Max_iter, lb, ub, dim, fobj):
    P_percent1 = 0.2
    P_percent2 = 0.4
    P_percent3 = 0.65
    pNum1 = round(N * P_percent1)
    pNum2 = round(N * P_percent2)
    pNum3 = round(N * P_percent3)

    Z = np.random.rand(N, dim)
    lambda_val = 0.4

    for i in range(N):
        for j in range(dim):
            if 0 < Z[i, j] <= (1 - lambda_val):
                Z[i, j] = Z[i, j] / (1 - lambda_val)
            else:
                Z[i, j] = (Z[i, j] - 1 + lambda_val) / lambda_val

    x = np.zeros((N, dim))
    fit = np.zeros(N)

    for i in range(N):
        x[i, :] = lb + (ub - lb) * Z[i, :]
        fit[i] = fobj(x[i, :])

    pFit = np.copy(fit)
    pX = np.copy(x)
    XX = np.copy(pX)
    fMin = np.min(fit)
    bestX = x[np.argmin(fit), :]

    for t in range(Max_iter):
        Wmax = 0.9
        Wmin = 0.782
        r1 = (Wmax - Wmin) / 2 * np.cos(np.pi * t / Max_iter) + (Wmax + Wmin) / 2
        w = Wmax - (Wmax - Wmin) * (t / Max_iter)

        worse = x[np.argmax(fit), :]

        for i in range(pNum1):
            if np.random.rand() < 0.5:
                b = 0.3
                k = 0.1
                r4 = np.random.rand()
                miu = 0.1
                a = 1 if r4 > miu else -1
                x[i, :] = pX[i, :] + b * abs(pX[i, :] - worse) + a * k * (XX[i, :])
            else:
                for j in range(dim):
                    r2 = (2 * np.pi) * np.random.rand()
                    r3 = 2 * np.random.rand()
                    x[i, j] = w * x[i, j] + (r1 * np.sin(r2) * (r3 * bestX[j] - x[i, j]))

            x[i, :] = bounds(x[i, :], lb, ub)
            fit[i] = fobj(x[i, :])

        bestXX = x[np.argmin(fit), :]
        R = 1 - t / Max_iter

        Xnew1 = bestXX * (1 - R)
        Xnew2 = bestXX * (1 + R)
        Xnew1 = bounds(Xnew1, lb, ub)
        Xnew2 = bounds(Xnew2, lb, ub)

        Xnew11 = bestX * (1 - R)
        Xnew22 = bestX * (1 + R)
        lbnew2 = bounds(Xnew11, lb, ub)
        ubnew2 = bounds(Xnew22, lb, ub)

        for i in range(pNum1, pNum2):
            x[i, :] = bestXX + (np.random.rand(1, dim) * (pX[i, :] - Xnew1) +
                                np.random.rand(1, dim) * (pX[i, :] - Xnew2))
            x[i, :] = bounds(x[i, :], Xnew1, Xnew2)
            fit[i] = fobj(x[i, :])

        for i in range(pNum2, pNum3):
            x[i, :] = pX[i, :] + (np.random.randn(1) * (pX[i, :] - lbnew2) +
                                  np.random.rand(1, dim) * (pX[i, :] - ubnew2))
            x[i, :] = bounds(x[i, :], lb, ub)
            fit[i] = fobj(x[i, :])

        s = 1
        for j in range(pNum3, N):
            x[j, :] = bestX + s * np.random.randn(1, dim) * ((abs(pX[j, :] - bestXX)) +
                                                             abs(pX[j, :] - bestX)) / 2
            x[j, :] = bounds(x[j, :], lb, ub)
            fit[j] = fobj(x[j, :])

        XX = np.copy(pX)
        for i in range(N):
            if fit[i] < pFit[i]:
                pFit[i] = fit[i]
                pX[i, :] = x[i, :]

            w1 = t / Max_iter
            w2 = 1 - t / Max_iter
            x[i, :] = pX[i, :] * (1 + w1 * np.random.randn() + w2 * np.tan((np.random.rand() - 1 / 2) * np.pi))
            x[i, :] = bounds(x[i, :], lb, ub)
            fit[i] = fobj(x[i, :])

            if fit[i] < pFit[i]:
                pFit[i] = fit[i]
                pX[i, :] = x[i, :]

            if pFit[i] < fMin:
                fMin = pFit[i]
                bestX = pX[i, :]

    return fMin, bestX

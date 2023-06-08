import math
import random as rn
import numpy as np
from sklearn.svm import NuSVR,SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.cluster import KMeans

class BSOAlgorithm:
    def sigmoid(x):
        return (1/(1 + math.exp(-x)))

    def svrCheck(X_train, y_train, X_val, y_val, sol):
#         clf = NuSVR(kernel = 'linear', gamma = 'auto', C = sol[0], nu = sol[1])
#         clf = SVR(kernel = 'linear', C = sol[0], epsilon = sol[1])
        clf = SVR(kernel='rbf', C=sol[0], gamma=sol[1], epsilon=sol[2])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        return (mse(y_val, y_pred))

    # Brain Storm Optimization,BSO,脑风暴优化算法

    def randGenSol(n, d = 2):
        S = []
        for i in range(n):
            l = []
            l.append(rn.uniform(1,10))
            l.append(rn.random())
            S.append(l)
    #     S = np.random.rand(n, d)
        return S

    def clustProbGen(clus, n, m):
        clus = list(clus)
        uc = list(set(clus))
        p = [1/len(uc)]*len(uc)
        for i in range(m):
            p[i] = clus.count(i)/n
        return p

    def probCheck(p):
        r = rn.random()
        if (r < p):
            return True
        return False

    def stepFun(t, T, k = 1):
        x = ((0.5 * T- t)/k)
        res = BSOAlgorithm.sigmoid(x) * rn.uniform(0,1)
        return res

    def genNewSol(x, t, T):
        n = len(x)
        y = [0]*n
        for i in range(n):
            y[i] = x[i] + BSOAlgorithm.stepFun(t, T) * rn.uniform(0,1)
        if(y[1] > 1):
            y[1] = BSOAlgorithm.sigmoid(y[1])
        return y

    def combineTwoSol(x1, x2):
        n = len(x1)
        x = [0]*n
        r = rn.random()
        for i in range(n):
            x[i] = (r * x1[i]) + ((1-r) * x2[i])
        return x

    def selClustCenters(X_train, y_train, X_val, y_val, S, lab, m):
        err = []
        cC = [[0,0]]*m
        cE = [9999999]*m
        best = 0
        for i in S:
                err.append(BSOAlgorithm.svrCheck(X_train, y_train, X_val, y_val, i))
        j = 0
        for i in lab:
            if(err[j] < cE[i]):
                cE[i] = err[j]
                cC[i] = j
            if(cE[i] < cE[best]):
                best = i
            j += 1
        return cC, best

    def bso(X_train, y_train, X_val, y_val, n, m):
        '''
            参数：
            X_train: 训练集特征数据
            y_train: 训练集目标变量数据
            X_val: 验证集特征数据
            y_val: 验证集目标变量数据
            n: 解的数量
            m: 聚类的数量
            返回值：
            最佳聚类中心对应的解
        '''
        pClustering = 0.5 # 选择1个聚类的概率
        pGeneration = 0.5 # 选择2个聚类的概率，pGeneration = 1 - pClustering
        pOneCluster = 0.5 # 选择1个聚类中聚类中心的概率
        pTwoCLuster = 0.5 # 选择2个聚类中聚类中心的概率
        if(m == 1):
            pGeneration = 1

        T = 10 # 最大迭代次数

        Solutions = BSOAlgorithm.randGenSol(n) #随机生成的解,初始化 Solutions
        for t in range(T):
            clust = KMeans(n_clusters=m,max_iter=10,n_init=10,init="k-means++",
                 algorithm="elkan",tol=1e-4,random_state=0).fit(Solutions) # 使用 K-means 聚类算法对 Solutions 进行聚类，得到 clust 对象
    #         clust = KMeans(n_clusters = m, random_state = 0).fit(Solutions) 
            #clust = AgglomerativeClustering(n_clusters = m).fit(Solutions) 
            prob = BSOAlgorithm.clustProbGen(clust.labels_, n, m) # 根据聚类结果计算每个解属于每个聚类的概率，得到 prob 对象
            #print(clust.labels_)
            cCenters, best = BSOAlgorithm.selClustCenters(X_train, y_train, X_val, y_val, Solutions, clust.labels_, m) # 根据聚类结果和概率选择聚类中心和最佳解
            #print(svrCheck(X_train, y_train, X_val, y_val, Solutions[cCenters[best]]))
            if(BSOAlgorithm.probCheck(pClustering)):    # 如果满足聚类的概率 pClustering，随机选择一个聚类中心，用新生成的解替换该聚类中心的解
                index = rn.choice(cCenters)
                new = BSOAlgorithm.randGenSol(1)[0]      
                Solutions[index] = new
            newSols = [] # 根据概率 pGeneration，选择生成解的方式
            for i in range(n):
                flag = 0
                if(BSOAlgorithm.probCheck(pGeneration)): 
                    flag = 1
                    selCluster = rn.choices(range(m), prob)[0]
                    if(BSOAlgorithm.probCheck(pOneCluster)): # 如果满足 pOneCluster，从单个聚类中随机选择一个解，生成新的解，并替换原解
                        #print("Case-1")
                        index = cCenters[selCluster]
                        new = BSOAlgorithm.genNewSol(np.array(Solutions)[index], t, T)
                    else: # 如果不满足 pOneCluster，从两个不同的聚类中各选择一个解，进行组合生成新的解，并替换原解
                        #print("Case-2")
                        sel = list(rn.choice(np.array(Solutions)[clust.labels_ == selCluster]))
                        new = BSOAlgorithm.genNewSol(sel, t, T)
                        index = Solutions.index(sel)
                else: # 根据生成的解和原解的性能进行比较，如果新解的性能更好，则更新原解
                    flag = 2
                    selCluster1, selCluster2 = rn.choices(range(m), prob, k = 2)
                    if(BSOAlgorithm.probCheck(pTwoCLuster)):
                        #print("Case-3")
                        index1 = cCenters[selCluster1]
                        index2 = cCenters[selCluster2]
                        comb = BSOAlgorithm.combineTwoSol(np.array(Solutions)[index1], Solutions[index2])
                        new = BSOAlgorithm.genNewSol(comb, t, T)
                    else:
                        #print("Case-4")
                        sel1 = list(rn.choice(np.array(Solutions)[clust.labels_ == selCluster1]))
                        sel2 = list(rn.choice(np.array(Solutions)[clust.labels_ == selCluster2]))
                        comb = BSOAlgorithm.combineTwoSol(sel1, sel2)
                        new = BSOAlgorithm.genNewSol(comb, t, T)
                        index1 = Solutions.index(sel1)
                        index2 = Solutions.index(sel2)
                if(flag == 1):
                    if(BSOAlgorithm.svrCheck(X_train, y_train, X_val, y_val, new) < BSOAlgorithm.svrCheck(X_train, y_train, X_val, y_val, Solutions[index])):
                        Solutions[index] = new
                elif(flag == 2):
                    if(BSOAlgorithm.svrCheck(X_train, y_train, X_val, y_val, new) < BSOAlgorithm.svrCheck(X_train, y_train, X_val, y_val, Solutions[index1])):
                        Solutions[index1] = new
                    elif(BSOAlgorithm.svrCheck(X_train, y_train, X_val, y_val, new) < BSOAlgorithm.svrCheck(X_train, y_train, X_val, y_val, Solutions[index2])):
                        Solutions[index2] = new 

        cCenters, best = BSOAlgorithm.selClustCenters(X_train, y_train, X_val, y_val, Solutions, clust.labels_, m) # 选择最佳聚类中心和对应的解
        print("Validation MSE:", BSOAlgorithm.svrCheck(X_train, y_train, X_val, y_val, Solutions[cCenters[best]])) # 打印验证集上的均方误差（MSE）结果
        return Solutions[cCenters[best]] # 返回最佳聚类中心对应的解
    
    # 不同k值对模型性能的影响
    def checkK(n):
        k = list(range(1,n))
        E = []
        for i in k:
            best = BSOAlgorithm.bso(X_train, y_train, X_val, y_val, 10, i)
            print("Parameters :", best)
            E.append(BSOAlgorithm.svrCheck(X_train, y_train, X_test, y_test, best))
            print("Test MSE:", E[i-1])
        plt.plot(k,E)
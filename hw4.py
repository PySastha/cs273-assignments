# CS273A - HOMEWORK 4
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
import texttable as tt
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# Statement of Collaboration
print("Statement of Collaboration:")
print("\tI have solved this assignment without any collaboration with fellow classmates")
print("\tVisited blogs, websites for python syntax, coding, library info\n")
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
def find_stat_tabular(inp):
    list_min = []
    list_max = []
    list_mean = []
    list_var = []

    for i in range(14):
        list_min.append(np.amin(inp[:, [i]]))
        list_max.append(np.amax(inp[:, [i]]))
        list_mean.append(np.mean(inp[:, [i]]))
        list_var.append(np.var(inp[:, [i]]))

    tab = tt.Texttable()
    headings = ['Feature', 'Min', 'Max', 'Mean', 'Var']
    tab.header(headings)

    for i in zip(range(1, 15), list_min, list_max, list_mean, list_var):
        tab.add_row(i)
    print(tab.draw())
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# Loading data:
X = np.genfromtxt('hw4_data/X_train.txt', delimiter=None)
Y = np.genfromtxt('hw4_data/Y_train.txt', delimiter=None)
X,Y = ml.shuffleData(X,Y)

print("\nQ-1 Solution:\n-------------\n")
print("Statistics before Scaling")
#find_stat_tabular(X)

Xtr, Xva, Ytr, Yva = ml.splitData(X, Y)
Xt, Yt = Xtr[:5000], Ytr[:5000] # subsample for efficiency (you can go higher)
XtS, params = ml.rescale(Xt) # Normalized XtS
XvS, _ = ml.rescale(Xva,params) # Normalized XvS

print("Statistics after Scaling")
#find_stat_tabular(XtS)
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
def linear_auc_plot(xt,xv,yt,yv):
    list_tr = []
    list_va = []
    r = [-10, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5, 10]

    for i in r:
        learner = ml.linearC.linearClassify()
        learner.train(xt, yt, reg=i, initStep=0.5, stopTol=1e-6, stopIter=100)
        print("running", i)

        temp1 = learner.auc(xt, yt)
        list_tr.append(temp1)
        temp2 = learner.auc(xv, yv)
        list_va.append(temp2)

    plt.plot(r, list_tr)
    plt.plot(r, list_va)
    plt.show()
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
print("\nQ-2 Solution:\n-------------\n")
print("AUC value is the area under ROC curve")
print("Initially, reg was swept from -10 to +10. \nSince it is changing only between -2 to +5, that area has small step size")

#linear_auc_plot(XtS,XvS,Yt,Yva)

#XtS_d2 = ml.transforms.fpoly(Xt,2,False) # Degree 2
#XtS_d2,params = ml.rescale(XtS_d2)
#XvS_d2 = ml.transforms.fpoly(Xva,2,False) # Degree 2
#XvS_d2, _ = ml.rescale(XvS_d2,params) # Normalize the features

#print("\n", XtS_d2) # To have a idea about degree 2 poly
#print(XtS_d2.shape)
#print("It contains all combinations (2nd degree) --> 14 features, their squares and combinations of 2 features")

#linear_auc_plot(XtS_d2,XvS_d2,Yt,Yva)
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
def nn_auc_plot(xt,xv,yt,yv):
    list_tr = []
    list_va = []
    r = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]

    for i in r:
        learner = ml.knn.knnClassify()
        learner.train(xt, yt, K=i, alpha=0.0)
        print("running", i)

        temp1 = learner.auc(xt, yt)  # train AUC
        list_tr.append(temp1)
        temp2 = learner.auc(xv, yv)  # train AUC
        list_va.append(temp2)

    plt.plot(r, list_tr)
    plt.plot(r, list_va)
    plt.show()
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
print("\nQ-3 Solution:\n-------------\n")

#nn_auc_plot(XtS,XvS,Yt,Yva)
#nn_auc_plot(Xt,Xva,Yt,Yva)

run_nn = 0
if run_nn:
    K = range(1, 10, 1)  # Or something else
    A = range(0, 5, 1)  # Or something else
    tr_auc = np.zeros((len(K), len(A)))
    va_auc = np.zeros((len(K), len(A)))

    for i, k in enumerate(K):
        for j, a in enumerate(A):
            print(i, j)
            learner = ml.knn.knnClassify()
            learner.train(XtS, Yt, K=k, alpha=a)
            tr_auc[i][j] = learner.auc(XtS, Yt)  # train AUC
            va_auc[i][j] = learner.auc(XvS, Yva)  # train AUC

    A = list(A)
    K = list(K)

    # Now plot it
    f, ax = plt.subplots(1, 1, figsize=(8, 5))
    cax = ax.matshow(tr_auc, interpolation='nearest')
    f.colorbar(cax)
    ax.set_xticklabels([''] + A)
    ax.set_yticklabels([''] + K)
    plt.show()

    f, ax = plt.subplots(1, 1, figsize=(8, 5))
    cax = ax.matshow(va_auc, interpolation='nearest')
    f.colorbar(cax)
    ax.set_xticklabels([''] + A)
    ax.set_yticklabels([''] + K)
    plt.show()
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
def dtree_auc_plot(xt,xv,yt,yv):
    list_tr = []
    list_va = []
    list_nodes1 = []
    list_nodes2 = []
    list_nodes3 = []
    r = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for i in r:
        # for 4.1, 4.2(b)
        learner = ml.dtree.treeClassify()
        learner.train(xt, yt, maxDepth=i, minParent=2, minLeaf=1)
        # for 4.2 (b) - increase minParent
        learner2 = ml.dtree.treeClassify()
        learner2.train(xt, yt, maxDepth=i, minParent=4, minLeaf=1)
        # for 4.2 (b) - increase minLeaf
        learner3 = ml.dtree.treeClassify()
        learner3.train(xt, yt, maxDepth=i, minParent=2, minLeaf=2)
        print("running", i)

        # for 4.1
        #temp1 = learner.auc(xt, yt)  # train AUC
        #list_tr.append(temp1)
        #temp2 = learner.auc(xv, yv)  # train AUC
        #list_va.append(temp2)

        # for 4.2 (a)
        num_nodes = learner.sz
        list_nodes1.append(num_nodes)
        # for 4.2 (b)
        num_nodes = learner2.sz
        list_nodes2.append(num_nodes)
        # for 4.2 (b)
        num_nodes = learner3.sz
        list_nodes3.append(num_nodes)

    # for 4.1
    #plt.plot(r, list_tr)
    #plt.plot(r, list_va)

    # for 4.2 (a)
    plt.plot(r, list_nodes1, label = "maxDepth varied")
    plt.plot(r, list_nodes2, label = "maxDepth varied, mP incr")
    plt.plot(r, list_nodes3, label = "maxDepth varied, mL incr")
    plt.legend(loc='upper left')
    plt.show()
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
print("\nQ-4 Solution:\n-------------\n")

#dtree_auc_plot(Xt, Xva, Yt, Yva)

run_dt = 0
if run_dt:
    K = range(1, 10, 1)  # Or something else
    A = range(0, 5, 1)  # Or something else
    tr_auc = np.zeros((len(K), len(A)))
    va_auc = np.zeros((len(K), len(A)))

    for i, k in enumerate(K):
        for j, a in enumerate(A):
            print(i, j)

            learner = ml.dtree.treeClassify()
            learner.train(Xt, Yt, maxDepth=5, minParent=i, minLeaf=j)

            tr_auc[i][j] = learner.auc(XtS, Yt)  # train AUC
            va_auc[i][j] = learner.auc(XvS, Yva)  # train AUC

    A = list(A)
    K = list(K)

    # Now plot it
    f, ax = plt.subplots(1, 1, figsize=(8, 5))
    cax = ax.matshow(tr_auc, interpolation='nearest')
    f.colorbar(cax)
    ax.set_xticklabels([''] + A)
    ax.set_yticklabels([''] + K)
    plt.show()

    f, ax = plt.subplots(1, 1, figsize=(8, 5))
    cax = ax.matshow(va_auc, interpolation='nearest')
    f.colorbar(cax)
    ax.set_xticklabels([''] + A)
    ax.set_yticklabels([''] + K)
    plt.show()
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
print("\nQ-5 Solution:\n-------------\n")

run_nnet = 1
if run_nnet:
    K = range(1, 10, 1)  # Or something else
    A = range(0, 5, 1)  # Or something else
    tr_auc = np.zeros((len(K), len(A)))
    va_auc = np.zeros((len(K), len(A)))

    for i, k in enumerate(K):
        for j, a in enumerate(A):
            print(i, j)
            a = [14]    #no of features
            b = [i] * j
            c = [2]     #either rain = 0 or 1
            abc = np.concatenate([a, b, c])

            nn = ml.nnet.nnetClassify()
            nn.init_weights(abc, 'random', XtS, Yt)  # as many layers nodes you want
            nn.train(XtS, Yt, stopTol=1e-8, stepsize=.25, stopIter=300)

            tr_auc[i][j] = learner.auc(XtS, Yt)  # train AUC
            va_auc[i][j] = learner.auc(XvS, Yva)  # train AUC

    A = list(A)
    K = list(K)

    # Now plot it
    f, ax = plt.subplots(1, 1, figsize=(8, 5))
    cax = ax.matshow(tr_auc, interpolation='nearest')
    f.colorbar(cax)
    ax.set_xticklabels([''] + A)
    ax.set_yticklabels([''] + K)
    plt.show()

    f, ax = plt.subplots(1, 1, figsize=(8, 5))
    cax = ax.matshow(va_auc, interpolation='nearest')
    f.colorbar(cax)
    ax.set_xticklabels([''] + A)
    ax.set_yticklabels([''] + K)
    plt.show()
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----



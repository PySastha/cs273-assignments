# CS273A - HOMEWORK 4
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
import texttable as tt
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

#print("\nQ-1 Solution:\n-------------\n")
#print("Statistics before Scaling")
#find_stat_tabular(X)

Xtr, Xva, Ytr, Yva = ml.splitData(X, Y)
Xt, Yt = Xtr[:5000], Ytr[:5000] # subsample for efficiency (you can go higher)
XtS, params = ml.rescale(Xt) # Normalized XtS
XvS, _ = ml.rescale(Xva,params) # Normalized XvS

#print("Statistics after Scaling")
#find_stat_tabular(XtS)
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
def linear_auc_plot(xt,xv,yt,yv):
    list_tr = []
    list_va = []
    #r = [-10, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5, 10]
    r = [0,1,2,3,4]
    for i in r:
        learner = ml.linearC.linearClassify()
        learner.train(xt, yt, reg=i, initStep=0.5, stopTol=1e-6, stopIter=100)
        #print("running", i)

        temp1 = learner.auc(xt, yt)
        list_tr.append(temp1)
        temp2 = learner.auc(xv, yv)
        list_va.append(temp2)

    plt.plot(r, list_tr, label="tr")
    plt.plot(r, list_va, label="va)")
    plt.legend(loc='upper left')
    plt.show()
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
print("\nQ-2 Solution:\n-------------\n")
#print("AUC value is the area under ROC curve")
#print("Initially, reg was swept from -10 to +10. \nSince it is changing only between -2 to +5, that area has small step size")

#linear_auc_plot(XtS,XvS,Yt,Yva)

XtS_d2 = ml.transforms.fpoly(Xt,2,False) # Degree 2
XtS_d2,params = ml.rescale(XtS_d2)
XvS_d2 = ml.transforms.fpoly(Xva,2,False) # Degree 2
XvS_d2, _ = ml.rescale(XvS_d2,params) # Normalize the features

print("\n", XtS_d2) # To have a idea about degree 2 poly
print(XtS_d2.shape)

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
    K = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35]  # Or something else
    A = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35]  # Or something else
    tr_auc = np.zeros((len(K), len(A)))
    va_auc = np.zeros((len(K), len(A)))

    for i, k in enumerate(K):
        for j, a in enumerate(A):
            print(k, a)

            learner = ml.dtree.treeClassify(Xt, Yt, maxDepth=5, minParent=k, minLeaf=a)

            tr_auc[i][j] = learner.auc(Xt, Yt)  # train AUC
            va_auc[i][j] = learner.auc(Xva, Yva)  # train AUC

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
    K = range(1, 8)  # Or something else
    A = range(1, 4)  # Or something else
    tr_auc = np.zeros((len(K), len(A)))
    va_auc = np.zeros((len(K), len(A)))

    for i, k in enumerate(K):
        for j, a in enumerate(A):
            print(i, j)

            abc_list = [14]    #no of features
            b = [i] * j
            c = 2     #either rain = 0 or 1

            for i in b:
                abc_list.append(i)
            abc_list.append(c)
            print(abc_list)

            nn = ml.nnet.nnetClassify()
            nn.init_weights(abc_list, 'random', XtS, Yt)  # as many layers nodes you want
            nn.train(XtS, Yt, stopTol=1e-8, stepsize=.25, stopIter=300)

            tr_auc[i][j] = nn.auc(XtS, Yt)  # train AUC
            va_auc[i][j] = nn.auc(XvS, Yva)  # train AUC

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

0 0
[14, 2]
it 1 : Jsur = 0.4438629435600522, J01 = 0.3156
it 2 : Jsur = 0.42851752984153607, J01 = 0.3102
it 4 : Jsur = 0.4198031347966193, J01 = 0.308
it 8 : Jsur = 0.41467169862690145, J01 = 0.3064
it 16 : Jsur = 0.41211574903873677, J01 = 0.305
it 32 : Jsur = 0.411500219359509, J01 = 0.3044
it 64 : Jsur = 0.411440733503668, J01 = 0.3036
it 128 : Jsur = 0.4114253064667242, J01 = 0.3034
it 256 : Jsur = 0.41140621358172996, J01 = 0.3034
0 1
[14, 0, 2]
it 1 : Jsur = 0.45955264470926405, J01 = 0.3428
it 2 : Jsur = 0.4546817531188677, J01 = 0.3428
it 4 : Jsur = 0.45283279293471623, J01 = 0.3428
it 8 : Jsur = 0.4517066227968129, J01 = 0.3428
it 16 : Jsur = 0.4509488455934124, J01 = 0.3428
it 32 : Jsur = 0.4506622150166264, J01 = 0.3428
it 64 : Jsur = 0.4505945948038136, J01 = 0.3428
it 128 : Jsur = 0.45058037327986183, J01 = 0.3428
0 2
[14, 0, 0, 2]
it 1 : Jsur = 0.45955264470926405, J01 = 0.3428
it 2 : Jsur = 0.4546817531188677, J01 = 0.3428
it 4 : Jsur = 0.45283279293471623, J01 = 0.3428
it 8 : Jsur = 0.4517066227968129, J01 = 0.3428
it 16 : Jsur = 0.4509488455934124, J01 = 0.3428
it 32 : Jsur = 0.4506622150166264, J01 = 0.3428
it 64 : Jsur = 0.4505945948038136, J01 = 0.3428
it 128 : Jsur = 0.45058037327986183, J01 = 0.3428
1 0
[14, 2]
it 1 : Jsur = 0.4438622699898867, J01 = 0.3156
it 2 : Jsur = 0.4285174540608923, J01 = 0.3102
it 4 : Jsur = 0.4198031250705773, J01 = 0.308
it 8 : Jsur = 0.414671691384098, J01 = 0.3064
it 16 : Jsur = 0.41211574342067125, J01 = 0.305
it 32 : Jsur = 0.4115002162987803, J01 = 0.3044
it 64 : Jsur = 0.4114407320741874, J01 = 0.3036
it 128 : Jsur = 0.41142530582610753, J01 = 0.3034
it 256 : Jsur = 0.41140621329658333, J01 = 0.3034
1 1
[14, 1, 2]
it 1 : Jsur = 0.4326657326298147, J01 = 0.3306
it 2 : Jsur = 0.43089646366752116, J01 = 0.3394
it 4 : Jsur = 0.42833853183004467, J01 = 0.3428
it 8 : Jsur = 0.42663074942437523, J01 = 0.3428
it 16 : Jsur = 0.4257378052363128, J01 = 0.3428
it 32 : Jsur = 0.425524627897119, J01 = 0.3428
it 64 : Jsur = 0.4256527857543391, J01 = 0.3428
it 128 : Jsur = 0.42589699599004727, J01 = 0.3428
it 256 : Jsur = 0.4261270993873646, J01 = 0.3428
1 2
[14, 1, 1, 2]
it 1 : Jsur = 0.4595526452594787, J01 = 0.3428
it 2 : Jsur = 0.454681752935489, J01 = 0.3428
it 4 : Jsur = 0.45283279264212034, J01 = 0.3428
it 8 : Jsur = 0.4517066224091217, J01 = 0.3428
it 16 : Jsur = 0.45094884332530655, J01 = 0.3428
it 32 : Jsur = 0.45066220787839883, J01 = 0.3428
it 64 : Jsur = 0.45059457719208, J01 = 0.3428
it 128 : Jsur = 0.4505803225638844, J01 = 0.3428
it 256 : Jsur = 0.4505770153770614, J01 = 0.3428
2 0
[14, 2]
it 1 : Jsur = 0.4438605902550772, J01 = 0.3156
it 2 : Jsur = 0.4285171693230981, J01 = 0.3102
it 4 : Jsur = 0.41980309116597664, J01 = 0.308
it 8 : Jsur = 0.4146716747690648, J01 = 0.3064
it 16 : Jsur = 0.41211571756463256, J01 = 0.305
it 32 : Jsur = 0.4115001989089751, J01 = 0.3044
it 64 : Jsur = 0.4114407229666716, J01 = 0.3036
it 128 : Jsur = 0.4114253013827262, J01 = 0.3034
it 256 : Jsur = 0.41140621119792214, J01 = 0.3034
2 1
[14, 2, 2]
it 1 : Jsur = 0.43112819073643077, J01 = 0.3202
it 2 : Jsur = 0.42364344523052905, J01 = 0.3072
it 4 : Jsur = 0.4193496527246325, J01 = 0.305
it 8 : Jsur = 0.4171356287404833, J01 = 0.3054
it 16 : Jsur = 0.41585763147417165, J01 = 0.3078
it 32 : Jsur = 0.41527775132515643, J01 = 0.309
it 64 : Jsur = 0.4152061831229115, J01 = 0.3096
it 128 : Jsur = 0.41545130195176444, J01 = 0.3092
it 256 : Jsur = 0.4158713895459052, J01 = 0.3108
2 2
[14, 2, 2, 2]
it 1 : Jsur = 0.45955264543444047, J01 = 0.3428
it 2 : Jsur = 0.4546817532170477, J01 = 0.3428
it 4 : Jsur = 0.4528327928345865, J01 = 0.3428
it 8 : Jsur = 0.4517066226567789, J01 = 0.3428
it 16 : Jsur = 0.450948844730527, J01 = 0.3428
it 32 : Jsur = 0.45066221222035135, J01 = 0.3428
it 64 : Jsur = 0.45059458876136393, J01 = 0.3428
it 128 : Jsur = 0.45058036031806503, J01 = 0.3428
3 0
[14, 2]
it 1 : Jsur = 0.4438619189490615, J01 = 0.3156
it 2 : Jsur = 0.42851736280867936, J01 = 0.3102
it 4 : Jsur = 0.41980310894786355, J01 = 0.308
it 8 : Jsur = 0.41467168401452675, J01 = 0.3064
it 16 : Jsur = 0.412115734639751, J01 = 0.305
it 32 : Jsur = 0.4115002106347807, J01 = 0.3044
it 64 : Jsur = 0.4114407291587856, J01 = 0.3036
it 128 : Jsur = 0.4114253044194422, J01 = 0.3034
it 256 : Jsur = 0.4114062126369204, J01 = 0.3034
3 1
[14, 3, 2]
it 1 : Jsur = 0.4267772247683465, J01 = 0.3058
it 2 : Jsur = 0.42140556055500983, J01 = 0.3064
it 4 : Jsur = 0.41782195281578827, J01 = 0.3076
it 8 : Jsur = 0.4145861960890041, J01 = 0.3078
it 16 : Jsur = 0.4121590726290448, J01 = 0.3058
it 32 : Jsur = 0.41023722156560405, J01 = 0.305
it 64 : Jsur = 0.4091144317189121, J01 = 0.3046
it 128 : Jsur = 0.4086466281435666, J01 = 0.3048
it 256 : Jsur = 0.40845356618949896, J01 = 0.3044
3 2
[14, 3, 3, 2]
it 1 : Jsur = 0.45955264506398497, J01 = 0.3428
it 2 : Jsur = 0.4546817531282952, J01 = 0.3428
it 4 : Jsur = 0.4528327929509137, J01 = 0.3428
it 8 : Jsur = 0.45170662280016777, J01 = 0.3428
it 16 : Jsur = 0.45094884559802606, J01 = 0.3428
it 32 : Jsur = 0.4506622150089399, J01 = 0.3428
it 64 : Jsur = 0.4505945947687047, J01 = 0.3428
it 128 : Jsur = 0.45058037319548094, J01 = 0.3428
4 0
[14, 2]
it 1 : Jsur = 0.44386186356611435, J01 = 0.3156
it 2 : Jsur = 0.4285174315800301, J01 = 0.3102
it 4 : Jsur = 0.4198031248841142, J01 = 0.308
it 8 : Jsur = 0.4146716909287423, J01 = 0.3064
it 16 : Jsur = 0.4121157429714579, J01 = 0.305
it 32 : Jsur = 0.41150021605343445, J01 = 0.3044
it 64 : Jsur = 0.41144073196255687, J01 = 0.3036
it 128 : Jsur = 0.4114253057776643, J01 = 0.3034
it 256 : Jsur = 0.4114062132756024, J01 = 0.3034
4 1
[14, 4, 2]
it 1 : Jsur = 0.4259940495981926, J01 = 0.306
it 2 : Jsur = 0.42225176653527813, J01 = 0.3054
it 4 : Jsur = 0.4187238299203742, J01 = 0.306
it 8 : Jsur = 0.41506918447954916, J01 = 0.3062
it 16 : Jsur = 0.4118568645866126, J01 = 0.3038
it 32 : Jsur = 0.40773109981120303, J01 = 0.3046
it 64 : Jsur = 0.4059452145607792, J01 = 0.3038
it 128 : Jsur = 0.40526883352694554, J01 = 0.3036
it 256 : Jsur = 0.4050430533533166, J01 = 0.3048
4 2
[14, 4, 4, 2]
it 1 : Jsur = 0.45955264607301377, J01 = 0.3428
it 2 : Jsur = 0.4546817530948294, J01 = 0.3428
it 4 : Jsur = 0.45283279288194683, J01 = 0.3428
it 8 : Jsur = 0.4517066227215341, J01 = 0.3428
it 16 : Jsur = 0.4509488452301083, J01 = 0.3428
it 32 : Jsur = 0.4506622139453539, J01 = 0.3428
it 64 : Jsur = 0.4505945926828399, J01 = 0.3428
it 128 : Jsur = 0.45058036933387574, J01 = 0.3428
5 0
[14, 2]
it 1 : Jsur = 0.4438632090389287, J01 = 0.3156
it 2 : Jsur = 0.42851758922296085, J01 = 0.3102
it 4 : Jsur = 0.4198031499208142, J01 = 0.308
it 8 : Jsur = 0.4146717064129175, J01 = 0.3064
it 16 : Jsur = 0.412115753865408, J01 = 0.305
it 32 : Jsur = 0.41150022184140417, J01 = 0.3044
it 64 : Jsur = 0.411440734631658, J01 = 0.3036
it 128 : Jsur = 0.41142530696254426, J01 = 0.3034
it 256 : Jsur = 0.41140621379936165, J01 = 0.3034
5 1
[14, 5, 2]
it 1 : Jsur = 0.42686394990281334, J01 = 0.3062
it 2 : Jsur = 0.42120749720817496, J01 = 0.3042
it 4 : Jsur = 0.41794596009768914, J01 = 0.306
it 8 : Jsur = 0.41444966006246303, J01 = 0.3058
it 16 : Jsur = 0.4112774627569457, J01 = 0.305
it 32 : Jsur = 0.4083214498945423, J01 = 0.3036
it 64 : Jsur = 0.40614798116790635, J01 = 0.3012
it 128 : Jsur = 0.405051604119777, J01 = 0.3012
it 256 : Jsur = 0.404676172790903, J01 = 0.3022
5 2
[14, 5, 5, 2]
it 1 : Jsur = 0.45955264551255604, J01 = 0.3428
it 2 : Jsur = 0.4546817533980983, J01 = 0.3428
it 4 : Jsur = 0.4528327928185065, J01 = 0.3428
it 8 : Jsur = 0.4517066225292775, J01 = 0.3428
it 16 : Jsur = 0.4509488439919106, J01 = 0.3428
it 32 : Jsur = 0.4506622087402851, J01 = 0.3428
it 64 : Jsur = 0.4505945770743305, J01 = 0.3428
it 128 : Jsur = 0.45058031441130225, J01 = 0.3428
it 256 : Jsur = 0.45057690368732456, J01 = 0.3428
6 0
[14, 2]
it 1 : Jsur = 0.4438619062122081, J01 = 0.3156
it 2 : Jsur = 0.42851740105689384, J01 = 0.3102
it 4 : Jsur = 0.4198031298256773, J01 = 0.308
it 8 : Jsur = 0.41467169518667524, J01 = 0.3064
it 16 : Jsur = 0.41211573815734, J01 = 0.305
it 32 : Jsur = 0.411500211551356, J01 = 0.3044
it 64 : Jsur = 0.41144072931500103, J01 = 0.3036
it 128 : Jsur = 0.41142530439331315, J01 = 0.3034
it 256 : Jsur = 0.41140621259354204, J01 = 0.3034
6 1
[14, 6, 2]
it 1 : Jsur = 0.427147877712612, J01 = 0.3052
it 2 : Jsur = 0.42117263190490356, J01 = 0.3046
it 4 : Jsur = 0.41751032839155167, J01 = 0.3064
it 8 : Jsur = 0.41386622851676985, J01 = 0.3052
it 16 : Jsur = 0.4106071155029331, J01 = 0.3036
it 32 : Jsur = 0.40770803524433225, J01 = 0.3028
it 64 : Jsur = 0.4052902764223248, J01 = 0.3018
it 128 : Jsur = 0.40297206999911866, J01 = 0.3014
it 256 : Jsur = 0.4015644001598176, J01 = 0.3012
6 2
[14, 6, 6, 2]
it 1 : Jsur = 0.45955264989355354, J01 = 0.3428
it 2 : Jsur = 0.4546817540087748, J01 = 0.3428
it 4 : Jsur = 0.4528327893073416, J01 = 0.3428
it 8 : Jsur = 0.45170660703246185, J01 = 0.3428
it 16 : Jsur = 0.4509486998491603, J01 = 0.3428
it 32 : Jsur = 0.45065728139628647, J01 = 0.3428
it 64 : Jsur = 0.41086093143695623, J01 = 0.3006
it 128 : Jsur = 0.40529062024168616, J01 = 0.2992
it 256 : Jsur = 0.3999083971432903, J01 = 0.2972



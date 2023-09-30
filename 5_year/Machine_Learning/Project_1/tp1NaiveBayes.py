"""
Created on Wed Oct 12 15:45:49 2022

@author: Joao Palma 55414
         Ruben Belo 55967
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score


def load_data():
    data = np.loadtxt("TP1_train.tsv")
    data = shuffle(data)
    Yr = data[:,4]
    Xr = data[:,:4]
    data = np.loadtxt("TP1_test.tsv")
    data = shuffle(data)
    Yt = data[:,4]
    Xt = data[:,:4]
    
    scaler=StandardScaler()
    Xr = scaler.fit_transform(Xr)
    Xt=scaler.transform(Xt)
    return Yr, Xr, Yt, Xt

def GaussianNaiveBayes(Y_train, X_train, Y_test, X_test):

    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    error = 1-clf.score(X_test,Y_test)
    prediction = clf.predict(X_test)
    
    return error, prediction
    
def calc_fold(X_train,Y_train,train_ix,valid_ix, bandwith_value):
    
    #kde = KernelDensity(kernel= "gaussian", bandwidth=bandwith_value)
   
    X_train_set = X_train[train_ix]
    Y_train_set = Y_train[train_ix]
    
    xs_0 = X_train_set[Y_train_set == 0]
    xs_1 = X_train_set[Y_train_set == 1]
    
    C0_log = np.log(len(xs_0) / len(X_train_set))
    C1_log = np.log(len(xs_1) / len(X_train_set))
    
    probs0 = np.full(len(X_train),C0_log) 
    probs1 = np.full(len(X_train),C1_log)
    
    for feats in range(X_train.shape[1]):
        kde = KernelDensity(kernel= "gaussian", bandwidth=bandwith_value)
        kde.fit(xs_0[:,[feats]])
        probs0 += kde.score_samples(X_train[:,[feats]])
        
        kde = KernelDensity(kernel= "gaussian", bandwidth=bandwith_value)
        kde.fit(xs_1[:,[feats]])
        probs1 += kde.score_samples(X_train[:,[feats]])
        
    pred = np.zeros(len(X_train))
    
    for i in range(len(X_train)):
        if(probs0[i] < probs1[i]):
            #print("0 =" , np.exp(probs0[i]), "vs 1 =", np.exp(probs1[i]))
            pred[i] = 1
            
    train_error = 1 - accuracy_score(Y_train[train_ix], pred[train_ix])        
    valid_error = 1 - accuracy_score(Y_train[valid_ix], pred[valid_ix])  
    return train_error,valid_error


def nb_true_err(X_train,Y_train,X_test,Y_test, bandwith_value):
    
    xs_0 = X_train[Y_train == 0]
    xs_1 = X_train[Y_train == 1]
    
    C0_log = np.log(len(xs_0) / len(X_train))
    C1_log = np.log(len(xs_1) / len(X_train))
    
    probs0 = np.full(len(X_test),C0_log) 
    probs1 = np.full(len(X_test),C1_log)
    
    for feats in range(X_train.shape[1]):
        
        kde = KernelDensity(kernel= "gaussian", bandwidth=bandwith_value)
        kde.fit(xs_0[:,[feats]])
        probs0 += kde.score_samples(X_test[:,[feats]])
        
        kde = KernelDensity(kernel= "gaussian", bandwidth=bandwith_value)
        kde.fit(xs_1[:,[feats]])
        probs1 += kde.score_samples(X_test[:,[feats]])
        
    pred = np.zeros(len(X_test))
    for i in range(len(X_test)):
        if(probs0[i] < probs1[i]):
            #print("0 =" , np.exp(probs0[i]), "vs 1 =", probs1[i])
            pred[i] = 1
    true_error = 1 - accuracy_score(Y_test, pred) 
    return true_error, pred, bandwith_value

def plot_nb(errors):
    plt.plot(errors[:,0], errors[:,1],color="red", label = "Validation error")
    plt.plot(errors[:,0], errors[:,2],color="blue", label = "Training error")
    plt.title("Naive Bayes Errors")
    plt.xlabel("Bandwith")
    plt.legend(loc=4)
    plt.savefig("NB.png", dpi=300)
    plt.show()
    
        
def naive_bayes(Y_train, X_train, Y_test, X_test): 
    
    folds=5
    bestVal=100000
    kf= StratifiedKFold(folds)
    errors=[]
    for b in range(2,62,2):
        bandwith_value=b/100
        tr_err = va_err = 0
        for train_ix,valid_ix in kf.split(Y_train,Y_train):
            r,v= calc_fold(X_train,Y_train,train_ix,valid_ix, bandwith_value)
            tr_err += r
            va_err += v
        #print(bandwith_value,':', tr_err/folds,va_err/folds)
        if bestVal>va_err/folds:
            bestVal = va_err/folds
            bestband = bandwith_value
        errors.append((bandwith_value,va_err/folds,tr_err/folds))
    plot_nb(np.array(errors))    
    return nb_true_err(X_train,Y_train,X_test,Y_test, bestband)
    
    
    
def normal_test(Y_test, prediction):
    size = len(Y_test) 
    errors = size - np.count_nonzero(Y_test == prediction)
    sigma =np.sqrt(size*(errors/size)*(1-errors/size))
    return errors-1.96*sigma, errors+1.96*sigma

def mcNemar(Y_test, prediction1,prediction2):
    e1=np.count_nonzero(np.logical_and(Y_test != prediction1,Y_test == prediction2))
    e2=np.count_nonzero(np.logical_and(Y_test != prediction2,Y_test == prediction1))
    return ((np.absolute(e1-e2)-1)**2)/(e1+e2)

def doNB():
    Y_train, X_train, Y_test, X_test = load_data()
    GNB_error,GBN_prediction = GaussianNaiveBayes(Y_train, X_train, Y_test, X_test)
    NB_error, NB_prediction,best_bandwith = naive_bayes(Y_train, X_train, Y_test, X_test)
    n1,n2 = normal_test(Y_test, GBN_prediction)
    
    print("Gaussian Naive Bayes:\n","True Error:",GNB_error,"\n",
          "Normal Test:",n1,",",n2,"\n")
    
    n1,n2 =normal_test(Y_test, NB_prediction)
    print("Naive Bayes:\n","Best Bandwith:",best_bandwith,"\n",
          "True Error:",NB_error,"\n","Normal Test:",n1,",",n2,"\n")
    
    print("Naive Bayes vs Gaussian Naive Bayes\n",
          "McNemar Test:",mcNemar(Y_test,NB_prediction,GBN_prediction),"\n")
    
doNB()
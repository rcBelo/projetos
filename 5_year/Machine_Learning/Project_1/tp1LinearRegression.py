"""
Created on Wed Oct 12 15:45:49 2022

@author: Joao Palma 55414
         Ruben Belo 55967
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle


def expand(data,degree):
     expanded = np.zeros((data.shape[0],degree+1))
     expanded[:,0]=data[:,0]
     expanded[:,-1]=data[:,-1]
     for power in range(2,degree+1):
         expanded[:,power-1]=data[:,0]**power
     return expanded
 
def loadData():
    df = np.loadtxt("SatelliteConjunctionDataRegression.csv",skiprows=1,delimiter=",")
    data = shuffle(df)
    
    Ys = data[:,6]
    Xs = data[:,:6]
    return Xs,Ys

def plot_values(Y_train, predict, train_ix, valid_ix, degree):
    """return classification error for train and test sets"""
 
    
    plt.scatter(Y_train[train_ix], predict[train_ix], marker = ".", label = "Train")
    
    plt.scatter(Y_train[valid_ix], predict[valid_ix],marker = ".", color = "red", label = "Test")
    
    x = np.linspace(min(Y_train),max(Y_train) , 100)
    plt.plot(x,x, color="black")
    
    plt.xlabel("true")
    plt.ylabel("predicted")
    plt.title("degree = " + str(degree))
    
    plt.legend(loc = "best")
    
    plt.savefig("REGRESS-PRED-VS-TRUE.png", dpi=300)
    
    plt.show()
    
def plot_valuesSave(Y_train, predict, train_ix, valid_ix, degree):
    """return classification error for train and test sets"""
 
    
    plt.scatter(Y_train[train_ix], predict[train_ix], marker = ".", label = "Train")
    
    plt.scatter(Y_train[valid_ix], predict[valid_ix],marker = ".", color = "red", label = "Test")
    
    x = np.linspace(min(Y_train),max(Y_train) , 100)
    plt.plot(x,x, color="black")
    
    plt.xlabel("true")
    plt.ylabel("predicted")
    plt.title("degree = " + str(degree))
    
    plt.legend(loc = "best")
    
    plt.savefig("REGRESS-PRED-VS-TRUE.png", dpi=300)
    
    plt.close() 
    
def plot_valid_VS_Train(error):
    error = np.array(error)
    plt.plot(error[:,0], error[:,1],color="red", label = "validation error", marker = "x")
    plt.plot(error[:,0], error[:,2],color="blue", label = "train error",marker = "s")
    plt.xlabel("degree")
    plt.legend(loc=2)
    plt.yscale('log')
    plt.title("Polynomial Linear Regression")
    plt.savefig("REGRESS-TR-VAL.png", dpi=300)
    plt.show()

def calc_fold(X,Y,train_ix,valid_ix):
    """return classification error for train and test sets"""
    reg = LinearRegression()
    reg.fit(X[train_ix,:],Y[train_ix])
    prob = reg.predict(X)
    squares = (prob-Y)**2
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix]), prob


def Standaradization(Xs, Ys):
    X_train,X_test,Y_train,Y_test = train_test_split(Xs, Ys , test_size=0.20)
    scaler=StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    Y_train = scaler.fit_transform(Y_train.reshape(-1,1))
    Y_test=scaler.transform(Y_test.reshape(-1,1))
    return X_train,X_test,Y_train,Y_test

def LRegression(X_train, X_test, Y_train, Y_test):
    error = []
    bestVal = 10000000
    bestDegree = 1
    folds = 10
    bestPredict = 0
    kf = KFold(n_splits=folds)
    for degree in range(1,7):
        poly = PolynomialFeatures(degree)
        sample = poly.fit_transform(X_train)
        tr_err = va_err = 0
        for train_ix,valid_ix in kf.split(sample):
            r,v, predict = calc_fold(sample,Y_train,train_ix,valid_ix)
            tr_err += r
            va_err += v
        #print(degree,':', tr_err/folds,va_err/folds)
        if bestVal>va_err/folds:
            bestVal = va_err/folds
            bestDegree = degree
            bestPredict = predict
        error.append((degree, va_err/folds, tr_err/folds))
        plot_values(Y_train, predict, train_ix, valid_ix, degree)
    plot_valuesSave(Y_train, bestPredict, train_ix, valid_ix, degree)
    plot_valid_VS_Train(error)
    return bestDegree
    
def LRegression_trueError(X_train,X_test,Y_train,Y_test, bestDegree):
    poly = PolynomialFeatures(bestDegree)
    sample = poly.fit_transform(X_train)
    reg = LinearRegression()
    reg.fit(sample,Y_train)
    sample = poly.fit_transform(X_test)
    return 1-reg.score(sample, Y_test)




def doLR():
    Xs, Ys = loadData()
    X_train,X_test,Y_train,Y_test = Standaradization(Xs, Ys)
    bestDegree = LRegression(X_train, X_test, Y_train, Y_test)
    true_error = LRegression_trueError(X_train, X_test, Y_train, Y_test, bestDegree)
    print("Polynomial Linear Regression:\n","Best Degree:",bestDegree,"\n",
          "True Error:",true_error,"\n")
    
    
doLR()
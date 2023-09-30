# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 15:10:44 2022

celulas diferentes classificar como similares

ou

celulas similares classificar como diferentes

@author: Asus
"""
from tp2_aux import images_as_matrix, report_clusters, report_clusters_hierarchical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, BisectingKMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import Normalizer

true_labels = np.loadtxt("labels.txt", delimiter=",", dtype = int)
true_labels = true_labels[true_labels[:,1] != 0]


def get_data(plot, scaling):
    data = images_as_matrix()

    # normal pca
    pca = PCA(n_components=6)
    pca.fit(data)
    pca_data = pca.transform(data)

    # kernel pca with rbf
    kpca = KernelPCA(n_components=6, kernel='rbf')
    kpca_data = kpca.fit_transform(data)

     # isomap
    iso = Isomap(n_components=6)
    iso_data = iso.fit_transform(data)

    pca_df = pd.DataFrame(
            pca_data, columns=['pca_A', 'pca_B', 'pca_C', 'pca_D', 'pca_E', 'pca_F',])
    #pd.plotting.scatter_matrix(pca_df,figsize=(50, 50))

    kpca_df = pd.DataFrame(kpca_data, columns=[
                               'kpca_A', 'kpca_B', 'kpca_C', 'kpca_D', 'kpca_E', 'kpca_F'])
    #pd.plotting.scatter_matrix(kpca_df,figsize=(50, 50))

    iso_df = pd.DataFrame(
            iso_data, columns=['iso_A', 'iso_B', 'iso_C', 'iso_D', 'iso_E', 'iso_F'])
    #pd.plotting.scatter_matrix(iso_df,figsize=(50, 50))

    frames = [pca_df, kpca_df, iso_df]
    df = pd.concat(frames, axis=1)
    
    if(plot == True):
        pd.plotting.scatter_matrix(pca_df,figsize=(50, 50))
        pd.plotting.scatter_matrix(kpca_df,figsize=(50, 50))
        pd.plotting.scatter_matrix(iso_df,figsize=(50, 50))
        pd.plotting.scatter_matrix(df, figsize=(50, 50))

    if(scaling):
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
    else:
        norm = Normalizer()
        df = norm.fit_transform(df)
        
    return df

def kmeans(df, rows, K):
    print("doing KMeans_" + str(K))
    kmeans = KMeans(n_clusters=K).fit(df)
    labels = kmeans.predict(df)
    labels = np.array(labels)
    report_clusters(rows, labels, "KMeans_" + str(K) + ".html")
    return getMetrics(labels, K)
    
def agglomerative(df, rows, K, link):
    print("doing Agglomerative_" + link + "_" + str(K))
    labels = AgglomerativeClustering(
        n_clusters=K, linkage = link).fit_predict(df)
    report_clusters(rows, labels, "Agglomerative_" + link + "_" + str(K) +"_.html")
    return getMetrics(labels, K)
    
def agglomerative_connectivity(df, rows, K, link, neighbors):
    print("doing Agglomerative_connectivity_" + str(neighbors) + "_" +  link + "_" + str(K))
    connectivity = kneighbors_graph(df, n_neighbors=neighbors, include_self=False)
    labels = AgglomerativeClustering(n_clusters=K, connectivity=connectivity,
                                     linkage = link).fit_predict(df)
    report_clusters(rows, labels, "Agglomerative_connectivity_" + str(neighbors) + "_" +  link + "_" + str(K) +"_.html")
    return getMetrics(labels, K)
    
def spectral(df, rows, K):
    print("doing spectral_" + str(K))
    labels = SpectralClustering(
        n_clusters=K, assign_labels="cluster_qr").fit_predict(df)

    report_clusters(rows, labels, "spectral_" + str(K) + ".html")
    return getMetrics(labels, K)

def spectralbi(df, rows, K):
    print("doing BisectingKMeans_" + str(K))
    labels = BisectingKMeans(
        n_clusters=K, bisecting_strategy='biggest_inertia').fit_predict(df)

    report_clusters(rows, labels, "BisectingKMeans_" + str(K) + ".html")
    return getMetrics(labels, K)
    
def bisecting_kmeans_hierarchical(df, rows, K):
    print("doing bisecting_kmeans_hierarchical_" + str(K))
    nr_clusters = 1

    X = []
    X.append(df)

    mapa = np.zeros((563, 2), dtype=int)
    mapa[:, 0] = range(563)
    mapa[:, 1] = range(563)

    where_is_data = []
    where_is_data.append(mapa)

    pred_labels = np.empty(len(rows), dtype=list)
    for idx in range(len(rows)):
            pred_labels[idx] = []

    s = []

    while nr_clusters != K:
            for x in X:
                s.append(len(x))

            the_chosen_one = np.argmax(s)
            data = X.pop(the_chosen_one)
            mapa = where_is_data.pop(the_chosen_one)

            kmeans = KMeans(n_clusters=2).fit(data)
            nr_clusters += 1
            labels = kmeans.predict(data)

            X.append(data[labels == 1])
            X.append(data[labels == 0])

            pool1 = mapa[labels == 1]
            pool0 = mapa[labels == 0]
            pool1[:, 0] = range(np.sum(labels))
            pool0[:, 0] = range(len(labels) - np.sum(labels))

            for value in pool1[:, 1]:
                pred_labels[value].append(1)
            for value in pool0[:, 1]:
                pred_labels[value].append(0)

            where_is_data.append(pool1)
            where_is_data.append(pool0)

            s = []
            
    report_clusters_hierarchical(rows, pred_labels, "bisecting_kmeans_hierarchical_" + str(K) + ".html")
    
    return getMetrics(transform_list_to_cluster(pred_labels, K), K)
    
def getMetrics(pred_labels, K):
    
    TP = TN = FP = FN = 0
    #print( len(true_labels[:,0]))
    for fst_row in range(0,len(true_labels[:,0])):
      for scd_row in  range(fst_row+1,len(true_labels[:,0])):
          if(true_labels[fst_row,1] == true_labels[scd_row,1] and 
             pred_labels[true_labels[fst_row,0]] == pred_labels[true_labels[scd_row,0]]):
              TP+=1
          if(true_labels[fst_row,1] != true_labels[scd_row,1] and 
             pred_labels[true_labels[fst_row,0]] != pred_labels[true_labels[scd_row,0]]):
              TN+=1
          if(true_labels[fst_row,1] == true_labels[scd_row,1] and 
             pred_labels[true_labels[fst_row,0]] != pred_labels[true_labels[scd_row,0]]):
              FN+=1
          if(true_labels[fst_row,1] != true_labels[scd_row,1] and 
             pred_labels[true_labels[fst_row,0]] == pred_labels[true_labels[scd_row,0]]):  
              FP+=1
            

    conf_mat=np.full((np.max(pred_labels)+1, np.max(true_labels[:,1])), 0, dtype=int)
    for idx, value in true_labels:
        conf_mat[pred_labels[idx]-1,value-1]+=1  
    purity_sums=0    
    for row in conf_mat:
        purity_sums+=max(row)
    total_purity=purity_sums/len(true_labels[:,0])
    
    recall=TP/(TP+FN)          
    precision=TP/(TP+FP)          
    Rand = (TP+TN)/(TP+FP+FN+TN)
    B = 0.5
    Fmesure=(B*B+1)*((precision*recall)/(B*B*precision+recall))
    print(recall, precision, Rand,total_purity, Fmesure)
    return K,recall,precision,Rand,total_purity,Fmesure

def createPlot1(data, T):
    df = pd.DataFrame(data,columns = ['NCluster','Recall','Precision','Rand','Purity','F0.5_score'])
    df.plot(kind="line",x="NCluster",xticks=df.NCluster, title = T, yticks = np.arange(0,1.1,0.1))
    
    
def transform_list_to_cluster(clusters_in_list, K):
    new_labels = [-1] * len(clusters_in_list)

    nr_cluster = 5

    i = 0
    for idx, value in enumerate(clusters_in_list):
        if(new_labels[idx] == -1):
            new_labels[idx] = i
            for x in range(idx+1, len(clusters_in_list)):
                if(value == clusters_in_list[x]):
                    new_labels[x] = i
            i+=1
    return new_labels

def createplot2(datas):
    dfs=[]
    for dt in datas:
        df2 = pd.DataFrame(dt,columns = ['NCluster','Recall','Precision','Rand','Purity','F0_5_score'])
        dfs.append(df2)
    fig = plt.figure() 
    fig.set_figwidth(10)
        
    fig.set_figheight(12)
    
    for x in range(0,4):
        plt.subplot(4,1, x+1)
        plt.plot(dfs[x].NCluster, dfs[x].Recall, color='b',label="Recall")
        plt.plot(dfs[x].NCluster, dfs[x].Precision, color='orange',label="Precision")
        plt.plot(dfs[x].NCluster, dfs[x].Purity, color='r',label="Purity")
        plt.plot(dfs[x].NCluster, dfs[x].Rand, color='green',label="Rand")   
        plt.plot(dfs[x].NCluster, dfs[x].F0_5_score, color='purple',label="F0.5_score")  
        #plt.xticks(range(3,np.max(dfs[x].NCluster)+1,1)) 
        plt.xticks(dfs[x].NCluster.astype(int))
        plt.grid(axis = 'x')  
        plt.tight_layout()
        if(x==0):
            plt.legend()
            plt.title('Different Linkages')
        if(x==3):
            plt.xlabel('Number of Clusters')
    

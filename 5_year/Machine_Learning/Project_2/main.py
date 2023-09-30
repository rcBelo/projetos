# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 15:41:28 2022

@author: Asus
"""

from tp2 import get_data, kmeans, agglomerative, agglomerative_connectivity, spectral, bisecting_kmeans_hierarchical, spectralbi
from tp2 import createPlot1, createplot2
import numpy as np

operations = ("Kmeans",
              "Agglomerative", 
              "Agglomerative/connectivity",
              "Spectral", 
              "hierarchical",
              "BisectingKMeans_")


"""operations = ("Kmeans",
              "Agglomerative", 
              "Agglomerative/connectivity",
              "Spectral", 
              "hierarchical")"""

min_cluster = 4
max_cluster = 9

link = ("ward",
        "complete", 
        "single",
        "average")

min_neighbors = 5

max_neighbors = 11

rows = np.arange(0,563)

"""change to false to normalize the data"""
scaling = True
"""change to false to normalize the data"""

df = get_data(True, scaling)

metrics = np.zeros((5,6))
agglo = []

r = 1

best_score = 0

best_config = ""

scores = {}

for x in range(0, r):
    print("iteração nddsndsndisnsind      " + str(x))
    for operation in operations:
        if operation == "Kmeans":
            for K in range(min_cluster,max_cluster):
                metrics[K - min_cluster] = kmeans(df, rows, K)
                score = metrics[K-min_cluster][3]*0.5 + metrics[K-min_cluster][5]*0.5
                if(score > best_score):
                    best_score = score
                    best_config = operation + " " +  str(K)
            createPlot1(metrics, "kmeans")
            
        if operation == "Agglomerative":
            for l in link:
                for K in range(min_cluster,max_cluster):
                    metrics[K - min_cluster] = agglomerative(df, rows, K, l)
                    score = metrics[K-min_cluster][3]*0.5 + metrics[K-min_cluster][5]*0.5
                    if(score > best_score):
                        best_score = score
                        best_config = operation + " " + l + " " + str(K)
                agglo.append(metrics)
            createplot2(agglo)
            
        if operation == "Agglomerative/connectivity":
            for l in link:
                for K in range(min_cluster,max_cluster):
                    metrics[K - min_cluster] = agglomerative_connectivity(df, rows, K, l, 5)
                    score = metrics[K-min_cluster][3]*0.5 + metrics[K-min_cluster][5]*0.5
                    if(score > best_score):
                        best_score = score
                        best_config = operation + " " + l + " " + str(K)
                createPlot1(metrics, "Agglomerative with connectivity " + l)
                
        if operation == "Spectral":
            for K in range(min_cluster,max_cluster):
                    metrics[K - min_cluster] = spectral(df, rows, K)
                    score = metrics[K-min_cluster][3]*0.5 + metrics[K-min_cluster][5]*0.5
                    if(score > best_score):
                        best_score = score
                        best_config = operation +  " " + str(K)
            createPlot1(metrics, "Spectral")
            
        if operation == "hierarchical":
            for K in range(min_cluster,max_cluster):
                    metrics[K - min_cluster] = bisecting_kmeans_hierarchical(df, rows, K)
                    score = metrics[K-min_cluster][3]*0.5 + metrics[K-min_cluster][5]*0.5
                    if(score > best_score):
                        best_score = score
                        best_config = operation +  " " + str(K)
            createPlot1(metrics, "bisecting_kmeans_hierarchical")
        if operation == "BisectingKMeans_":
            for K in range(min_cluster,max_cluster):
                    metrics[K - min_cluster] = spectralbi(df, rows, K)
                    score = metrics[K-min_cluster][3]*0.5 + metrics[K-min_cluster][5]*0.5
                    if(score > best_score):
                        best_score = score
                        best_config = operation +  " " + str(K)
            createPlot1(metrics, "BisectingKMeans")    
            
            
            
    if best_config in scores.keys():
        score = scores.get(best_config)
        score[0] = (score[0]*score[1] + best_score)/(score[1] + 1)
        score[1] += 1
    else:
        scores.update({best_config: [best_score, 1]})
        
        
        

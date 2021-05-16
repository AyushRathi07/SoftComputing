import pandas as pd 
import numpy as np
import random
import operator
import math
from array import array

### reading the input csv file 
df_full = pd.read_csv("iris_csv.csv")
columns = list(df_full.columns)
features = columns[:len(columns)-1]
class_labels = list(df_full[columns[-1]])
df = df_full[features]

# Number of Attributes
num_attr = len(df.columns)

# Number of Clusters to make
k = 3

# Maximum number of iterations
MAX_ITER = 100

# Number of data points
n = len(df)

# Fuzzy parameter
m = 2.00
NO_OF_TIMES = 21

# small constant to rectify divide by zero error
L = 0.001 

# Randomly initializing Membership Matrix
def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat

# Defining random seed set
def getSeedSet():
    seed_set = list()
    indexes = list()
    for j in range(k):
        arr = []
        r = random.randint(0,n-1)
        indexes.append(r)
        for col,data in df.items():
            arr.append(data[r])
        seed_set.append(arr)
    return seed_set, indexes

# Calculating cluster centers
def calculateClusterCenter(membership_mat):
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = list()
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, list(zip(*temp_num)))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers

# Updating membership matrix after choosing the seed set
def updateMembershipSeed(membership_mat, cluster_centers, seed_set_indexes):
    p = float(2/(m-1))
    for i in range(n):
        index = 0
        flag = 0
        for s in seed_set_indexes:
            if s == i:
                flag = 1
                for j in range(k):
                    membership_mat[i][j] = 0
                membership_mat[i][index] = 1
            index = index + 1
        if flag == 0:
            x = list(df.iloc[i])
            distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
            for j in range(k):
                den = sum([math.pow(float((distances[j] + L)/(distances[c] + L)), p) for c in range(k)])
                membership_mat[i][j] = float(1/den)       
    return membership_mat

# Updating membership matrix in further iterations
def updateMembershipValue(membership_mat, cluster_centers):
    p = float(2/(m-1))
    for i in range(n):
        x = list(df.iloc[i])
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float((distances[j]+L)/(distances[c] + L)), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)       
    return membership_mat

# Diving the dataset into labels depending upon the no of clusters
def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels

# Jaccard accuracy
def calculateAccuracy(labels):
    classobj = df_full.iloc[: , num_attr]
    y_true = classobj.values
    y_clust = {}
    for i in range(k):
        Dict = {}
        for j in range(n):
            if labels[j] == i:
                if y_true[j] in Dict.keys():
                    Dict[y_true[j]] = Dict[y_true[j]] + 1
                else:
                    Dict[y_true[j]] = 1
        Keymax = max(Dict, key=Dict.get)
        y_clust[i] = Keymax
    # print(y_clust)
    num = 0
    for i in range(n):
        if y_true[i] == y_clust[labels[i]]:
            num = num + 1
    return float(num/n)

# Fuzzy C Means 
def fuzzyCMeansClustering():
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    cluster_centers, seed_set_indexes = getSeedSet()
    print(seed_set_indexes)
    membership_mat = updateMembershipSeed(membership_mat, cluster_centers, seed_set_indexes)
    while curr <= MAX_ITER:
        if curr != 0:
            membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)
        cluster_centers = calculateClusterCenter(membership_mat)
        curr += 1
    
    # print(membership_mat)
    return cluster_labels, cluster_centers


labels = []
for i in range(1,NO_OF_TIMES):
    labels, centers = fuzzyCMeansClustering()
    print("Iteration " + str(i))
    print("Printing Cluster Centers")
    print(centers)
    print("Printing Dataset as Labels")
    print(labels)
    
jaccard = calculateAccuracy(labels)
print("Accuracy of clustering: ")
print(str(jaccard*100) + "%")




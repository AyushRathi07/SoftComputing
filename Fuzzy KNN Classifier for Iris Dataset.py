import operator
import pandas as pd

from sklearn.model_selection import train_test_split


headers = ['Column1', 'Column2', 'Column3', 'Column4', 'Class']
data = pd.read_csv("iris_csv.csv", header=None, names=headers)
attributes = ['Column1', 'Column2', 'Column3', 'Column4']
x = data[attributes]
y = data['Class']
averageAccuracy = 0
MAX_ITER = 20

def getU(y_train, rows):
    U = [[0 for i in range(rows)] for j in range(3)]
    for i in range(rows):
        if(y_train.iloc[i]=="Iris-setosa"):
            U[0][i]=1
            U[1][i]=0
            U[2][i]=0
        elif(y_train.iloc[i] == "Iris-versicolor"):
            U[1][i]=1
            U[0][i]=0
            U[2][i]=0
        else:
            U[2][i]=1
            U[0][i]=0
            U[1][i]=0
    return U

def getNewU(X_test, X_train, U):
    newU = [[0 for i in range(len(X_test))] for j in range(3)]
    for i in range(len(X_test)):
        dict = {}
        for j in range(len(X_train)):
            dist = 0
            for k in range(len(X_train.columns)):
                # print(X_train.iloc[j,k])
                # print(X_test.iloc[i,k])
                x1 = float(X_train.iloc[j,k])
                y1 = float(X_test.iloc[i,k])
                dist = dist + abs(x1-y1)
            dict[j] = dist
        sorted_tuples = sorted(dict.items(), key=operator.itemgetter(1))
        dict = {k: v for k, v in sorted_tuples}
        cnt = 0
        d = 0
        n1 = 0
        n2 = 0
        n3 = 0
        for key, value in dict.items():
            if(cnt>=5):
                break
            cnt = cnt + 1
            if(value==0):
                value = 1
            n1 = n1 + U[0][key]*(1/(value*value))
            n2 = n2 + U[1][key]*(1/(value*value))
            n3 = n3 + U[2][key]*(1/(value*value))
            d = d + (1/(value*value))
        newU[0][i] = n1/d
        newU[1][i] = n2/d
        newU[2][i] = n3/d
    print(newU)
    return getNewU

def KNN():
    for iterations in range(MAX_ITER):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        # print(X_train)
        rows = len(X_train)
        U = getU(y_train, rows)
        newU = getNewU(X_test, X_train, U)
        accuracy = 0
        for i in range(len(X_test)):
            if(y_test.iloc[i]=="Iris-setosa"):
                accuracy = accuracy + newU[0][i]
            elif(y_train.iloc[i] == "Iris-versicolor"):
                accuracy = accuracy + newU[1][i]
            else:
                accuracy = accuracy + newU[2][i]
        accuracy = accuracy/len(X_test)
        print(accuracy*100)
        averageAccuracy = averageAccuracy + accuracy

KNN()
print(averageAccuracy * 5)
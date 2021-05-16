import operator
import pandas as pd

from sklearn.model_selection import train_test_split


headers = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7','Column8', 'Column9', 'Column10', 'Column11', 'Column12', 'Column13', 'Column14','Column15', 'Column16', 'Column17', 'Column18', 'Column19', 'Column20', 'Column21','Column22', 'Column23', 'Column24', 'Column25', 'Column26', 'Column27', 'Column28','Column29', 'Column30', 'Column31', 'Column32', 'Column33', 'Column34', 'Class']
data = pd.read_csv("Ionosphere.csv", header=None, names=headers)
attributes = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7','Column8', 'Column9', 'Column10', 'Column11', 'Column12', 'Column13', 'Column14','Column15', 'Column16', 'Column17', 'Column18', 'Column19', 'Column20', 'Column21','Column22', 'Column23', 'Column24', 'Column25', 'Column26', 'Column27', 'Column28','Column29', 'Column30', 'Column31', 'Column32', 'Column33', 'Column34']
x = data[attributes]
y = data['Class']

def getU(y_train, rows):
    U = [[0 for i in range(rows)] for j in range(2)]
    for i in range(rows):
        if(y_train.iloc[i]=="g"):
            U[0][i]=1
            U[1][i]=0
        else:
            U[1][i]=1
            U[0][i]=0

def getNewU(X_test, X_train, U):
    newU = [[0 for i in range(len(X_test))] for j in range(2)]
    for i in range(len(X_test)):
        dict = {}
        for j in range(len(X_train)):
            dist = 0
            for k in range(len(X_train.columns)):
                dist = dist + abs(X_train.iloc[j,k]-X_test.iloc[i,k])
            dict[j] = dist
        sorted_tuples = sorted(dict.items(), key=operator.itemgetter(1))
        dict = {k: v for k, v in sorted_tuples}
        cnt = 0
        d = 0
        n1 = 0
        n2 = 0
        for key, value in dict.items():
            if(cnt>=5):
                break
            cnt = cnt + 1
            if(value==0):
                value = 1
            n1 = n1 + U[0][key]*(1/(value*value))
            n2 = n2 + U[1][key]*(1/(value*value))
            d = d + (1/(value*value))
        newU[0][i] = n1/d
        newU[1][i] = n2/d

averageAccuracy = 0

def KNN():
    for iterations in range(20):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        rows = len(X_train)
        U = getU(y_train, rows)
        newU = getNewU()
        accuracy = 0;
        for i in range(len(X_test)):
            if(y_test.iloc[i]=="g"):
                accuracy = accuracy + newU[0][i]
            else:
                accuracy = accuracy + newU[1][i]
        accuracy = accuracy/len(X_test)
        print(accuracy*100)
        averageAccuracy = averageAccuracy + accuracy

KNN()
print("Average accuracy: ", averageAccuracy/20 * 100, "%")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("iris_csv.csv", header=None, names=['Sepal Length',	'Sepal Width', 'Petal Length',	'Petal Width',	'Class'])

x = data[['Sepal Length',	'Sepal Width', 'Petal Length',	'Petal Width']]
x=np.array(x)
x[:5]

one_hot_encoder = OneHotEncoder(sparse=False)
y = data.Class
y = one_hot_encoder.fit_transform(np.array(y).reshape(-1, 1))
y[:5]
MAX_ITER = 20

def sigmoid_util(x):
  return 1 / (1 + np.exp(-x))

def calc_util(x):
  return np.multiply(x, 1-x)

def randomWeightInitialize(nodes):
  layers, weights = len(nodes), []
  for i in range(1, layers):
      w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)] for j in range(nodes[i])]
      weights.append(np.matrix(w))
  return weights

def forwardPropagation(x, weights, layers):
    activation, layerInput = [x], x
    for j in range(layers):
        outputVal = sigmoid_util(np.dot(layerInput, weights[j].T))
        activation.append(outputVal)
        layerInput = np.append(1, outputVal)
    return activation


def backPropagation(y, activation, weights, layers):
    outputFinal = activation[-1]
    error = np.matrix(y - outputFinal)
    for j in range(layers, 0, -1):
        currr = activation[j]
        if(j > 1):
            prev = np.append(1, activation[j-1])
        else:
            prev = activation[0]
        delta = np.multiply(error, calc_util(currr))
        weights[j-1] += lr * np.multiply(delta.T, prev)
        w = np.delete(weights[j-1], [0], axis=1) 
        error = np.dot(delta, w) 
    return weights

def adjustWeights(X, Y, lr, weights):
    layers = len(weights)
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x = np.matrix(np.append(1, x)) 
        activations = forwardPropagation(x, weights, layers)
        weights = backPropagation(y, activations, weights, layers)
    return weights

def MLP(X_train, Y_train, epochs=20, nodes=[], lr=0.13):
    hidden_layers = len(nodes) - 1
    weights = randomWeightInitialize(nodes)
    for epoch in range(1, epochs+1):
        weights = adjustWeights(X_train, Y_train, lr, weights)
        if(epoch %20  == 0):
            x=1
    return weights

def predictClass(item, weights):
    layers = len(weights)
    item = np.append(1, item)
    activation = forwardPropagation(item, weights, layers)
    outputFinal = activation[-1].A1
    index = findMaxActivation(outputFinal)
    y = [0 for i in range(len(outputFinal))]
    y[index] = 1  
    return y 

def findMaxActivation(output):
    m, index = output[0], 0
    for i in range(1, len(output)):
        if(output[i] > m):
            m, index = output[i], i
    return index

def calcAccuracy(X, Y, weights):
    correct = 0
    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        guess = predictClass(x, weights)
        if(y == guess):
            correct += 1
    return correct / len(X)

input_layer = len(x[0]) 
output_layer = len(y[0]) 

layers = [input_layer, 4, output_layer]
lr, epochs = 0.13, 100

results=[]
for i in range(MAX_ITER):
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.7,shuffle=True)
  weights = MLP(x_train, y_train, epochs=epochs, nodes=layers, lr=lr);
  results.append(calcAccuracy(x_test, y_test, weights))

print("Results \n" + str(results))
print("\n")
print(sum(results)/20*100)
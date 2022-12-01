import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





def featureScale(x):

    average = np.mean(x, axis=0)
    range = np.max(x, axis = 0) - np.min(x, axis = 0)
    return (x-average)/range

class neuronet():
    def __init__(self, inputSize, layers):
        self.m = inputSize
        self.n = 3
        self.weights = []


        layerSize = inputSize
        for layer in layers:

            self.weights.append(np.random.rand(layerSize, layer) - 0.5)
            layerSize = layer

        for weight in self.weights:
            print(weight.shape)
    def feed_forward(self, x):
        X = x
        XatStep = []
        print(self.n)
        for i in range(self.n):
            print(i)
            X = self.activation(np.dot(X, self.weights[i]))
            XatStep.append(X)
        #print(X)


        return self.sigmoidLayer(X), XatStep

    def activation(self, x):
        return x

    def predict(self, x):
        result = np.ones(len(x))
        for i,X in enumerate(x):
            result[i] *= self.feed_forward((X))

        return self.sigmoidLayer(result)


    def cost_function(self, y_pred,y): #Binary Cross entropy
        totalCost = -1* np.mean(y * np.log(y_pred) + (1-y) * np.log(1- y_pred))
        return totalCost
        pass


    def sigmoidLayer(self, x):
        return 1/(1 + np.exp(-1 * x))

    def devSigmoid(self,x):
        return np.exp(-1 * x)/np.square((1 + np.exp(-1 * x)))

    def devCost(self,y_pred, y):
        return y/y_pred - (1-y)/(1-y_pred)

    def devRelu(self, x):
        if x > 0:
            return 1
        if x <= 0:
            return 0
    def devZ(self, x):
        return x
    def devTheta(self, theta):
        return theta



    def backProp(self,x ,y_pred ,y):
        dC = self.devCost(y_pred, y) * self.devSigmoid(self.activation(x)) * self.devRelu(x) * self.devZ(x)
        print(dC)
        pass





df = pd.read_csv("./pulsar_data_train.csv")
df_np = df.to_numpy()
X_train = df.iloc[:,:-1].to_numpy()
y_train = df.iloc[:,-1].to_numpy()

#Clean data
X_train[np.isnan(X_train)] = 0




df.head().to_csv("./pulsarStarHead.csv")


network = neuronet(8, [4,3,1])

X_scaled_train = featureScale(X_train)
y_pred, XatTimeStep = network.feed_forward(X_scaled_train[0])
print(network.cost_function(y_pred, y_train[0]))
print(XatTimeStep)
print(network.backProp(XatTimeStep[0][0], y_pred, y_train[0]))

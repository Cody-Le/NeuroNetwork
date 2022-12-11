import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math





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

            self.weights.append(np.random.rand(layerSize, layer))
            layerSize = layer

    def feed_forward(self, x):
        X = x
        XatStep = []
        XatStep.append(X)
        for i in range(self.n):

            X = self.activation(np.dot(X, self.weights[i]))
            XatStep.append(X)
        #print(X)


        return self.sigmoidLayer(X), XatStep

    def activation(self, x):
        X = x
        X[x < 0] = 0
        return X


    def predict(self, x):
        result = np.ones(len(x)) * 0.5

        for i,X in enumerate(x):


            result[i] *= self.feed_forward(X)[0]

        return result


    def combineTheta(self, theta1, theta2, substract = False):
        if (substract):
            return [theta1[0] + theta2[0], theta1[1] + theta2[1], theta1[2] + theta2[2]]
        else:
            return [theta1[0] - theta2[0], theta1[1] - theta2[1], theta1[2] - theta2[2]]

    def thetaShape(self, theta):

        return [theta[0].shape,theta[1].shape,theta[2].shape]


    def multiplyTheta(self, theta, multiplier):
        return [multiplier* theta[0], multiplier* theta[1],multiplier* theta[2]]



    def training(self, x, y, episelon,batch, epochs):
        costOverIteration = np.zeros(epochs)
        for epoch in range(epochs):
            print(epoch)
            m = len(x)
            n = int(math.floor(m / batch))
            lastN = m - batch * n
            for o in range(1, n):  # Dividing everything into epoch of batches



                totalDeltaC = None
                for i, X in enumerate(x[batch * (o - 1): batch * o]):

                    y_pred, xatstep = self.feed_forward((X))
                    if totalDeltaC != None:
                        totalDeltaC = self.combineTheta(self.backProp(xatstep, y_pred, y[i]), totalDeltaC)

                    else:
                        totalDeltaC = self.backProp(xatstep, y_pred, y[i])

                learningFactor = self.multiplyTheta(totalDeltaC,
                                                    episelon)  # Applying in episelon into the the derivative of theta]

                self.weights = self.combineTheta(self.weights, learningFactor, substract=True)


            new_pred = self.predict(x)
            print(self.cost_function(new_pred, y))




    pass



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

        X = x
        X[x > 0] = 1
        X[x <= 0] = 0
        return X
    def devZ(self, x):

        return x
    def devTheta(self, theta):
        return np.transpose(theta)[0]



    def backProp(self,x ,y_pred ,y):

        dCda = self.devCost(y_pred, y) * self.devSigmoid(self.activation(x[-1])) * self.devRelu(x[-1])
        print(dCda)
        dC1 =  dCda * self.devZ(x[-2][:,np.newaxis])



        dCd2a = dCda * self.devTheta(self.weights[-1]) * self.devRelu(x[-2])
        dC2 = np.dot(x[-3][:,np.newaxis], dCd2a[np.newaxis,:])
        dCd3a =  np.sum(dCd2a) * self.devRelu(x[-3])
        dC3 = np.dot(x[-4][:,np.newaxis], dCd3a[np.newaxis,:])
        return [dC3, dC2, dC1]


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
y_pred, XatTimeStep = network.feed_forward(X_scaled_train[5])

#print(network.cost_function(y_pred, y_train[0]))
print(X_scaled_train[5], y_pred)
costDev = network.backProp(XatTimeStep, y_pred, y_train[0])
network.training(X_scaled_train[0:10], y_train[0:10], 0.4,3, 10)

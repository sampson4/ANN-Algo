#NAME: SAMSON FAKUNLE A
#MATRIC NO: CPE/16/7834

import numpy as np

# Define the input and target data (using bipolar inputs and targets)
X = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
Y = np.array([-1, -1, 1, -1])

# Define the ANDNOT function using boolean expressions
def andnot(input1, input2):
    return (input1 == 1 and input2 == 0)

# Define the Adalene class
class Adalene:
    def __init__(self,  initial_weight, input_size, learning_rate, bias):
        self.weights = np.full(input_size, initial_weight)
        self.bias = bias
        self.lr = learning_rate
        
    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
    
    def train(self, X, Y, epochs):
        for i in range(epochs):
            for j in range(X.shape[0]):
                y_pred = self.predict(X[j])
                error = Y[j] - y_pred
                self.weights += self.lr * error * X[j]
                self.bias += self.lr * error


# Test the Adalene algorithm on the ANDNOT gate
adalene = Adalene(initial_weight=0.2, input_size=2, learning_rate=0.2, bias=0.2)
adalene.train(X, Y, epochs=2)

print("Prediction for input [-1,-1]: ", adalene.predict(np.array([-1,-1])))
print("Prediction for input [-1,1]: ", adalene.predict(np.array([-1, 1])))
print("Prediction for input [1,-1]: ", adalene.predict(np.array([1,-1])))
print("Prediction for input [1,1]: ", adalene.predict(np.array([1,1])))

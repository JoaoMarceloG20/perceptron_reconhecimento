import numpy as np

X = np.array([
    [1, -1, 1, -1, 1],
    [-1, 1, -1, 1, -1],
    [1, 1, 1, 1, 1],
    [-1, 1, -1, 1, -1],
    [1, -1, 1, -1, 1]
])

T = np.array([
    [1, 1, 1, 1, 1],
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1]
])

X_vector = X.flatten()
T_vector = T.flatten()

training_data = np.array([X_vector, T_vector])
labels = np.array([1, -1])  

weights = np.zeros(X_vector.shape)
bias = 0
learning_rate = 0.1
epochs = 10

def step_function(x):
    return 1 if x >= 0 else -1

for epoch in range(epochs):
    for i in range(len(training_data)):
        x_i = training_data[i]
        y_hat = step_function(np.dot(x_i, weights) + bias)
        error = labels[i] - y_hat
        weights += learning_rate * error * x_i
        bias += learning_rate * error

def test_perceptron(input_vector):
    result = step_function(np.dot(input_vector, weights) + bias)
    return result  

resultado_X = test_perceptron(X_vector)
print("Resultado para X:", resultado_X)  

resultado_T = test_perceptron(T_vector)
print("Resultado para T:", resultado_T)  
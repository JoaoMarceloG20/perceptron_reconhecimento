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
labels = np.array([
    [1, -1],  
    [-1, 1]   
])

weights = np.zeros((2, X_vector.size))  
bias = np.zeros(2)
learning_rate = 0.1
epochs = 1000

def step_function(x):
    return 1 if x >= 0 else -1

for epoch in range(epochs):
    for i in range(len(training_data)):
        x_i = training_data[i]
        for j in range(2): 
            y_hat = step_function(np.dot(x_i, weights[j]) + bias[j])
            error = labels[j][i] - y_hat
            weights[j] += learning_rate * error * x_i
            bias[j] += learning_rate * error

def test_perceptron(input_vector):
    results = []
    for j in range(2):  # Testando cada neurônio separadamente
        result = step_function(np.dot(input_vector, weights[j]) + bias[j])
        results.append(result)
    return results

resultado_X = test_perceptron(X_vector)
print("Resultados para X:", resultado_X)  
resultado_T = test_perceptron(T_vector)
print("Resultados para T:", resultado_T)  

new_letter = np.array([
    [1, 1, 1, 1, 1],
    [1, -1, -1, -1, 1],
    [1, 1, 1, 1, 1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1]
]).flatten()

resultado_A = test_perceptron(new_letter)
print("Resultados para uma nova letra:", resultado_A)
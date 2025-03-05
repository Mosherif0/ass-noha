import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

np.random.seed(1)
weights_input_hidden = np.random.uniform(-1, 1, (2, 2))  # أوزان الطبقة المخفية
weights_hidden_output = np.random.uniform(-1, 1, (2, 1))  # أوزان الطبقة النهائية
bias_hidden = np.random.uniform(-1, 1, (1, 2))  # انحياز الطبقة المخفية
bias_output = np.random.uniform(-1, 1, (1, 1))  # انحياز الطبقة النهائية


learning_rate = 0.5


epochs = 10000

def forward_propagation(X):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_layer_input)
    return hidden_layer_input, hidden_layer_output, final_layer_input, final_output

def backpropagation(X, Y, hidden_layer_output, final_output):
    error = Y - final_output
    d_output = error * sigmoid_derivative(final_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    global weights_hidden_output, weights_input_hidden, bias_output, bias_hidden
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    return np.mean(np.abs(error))

for epoch in range(epochs):
    hidden_layer_input, hidden_layer_output, final_layer_input, final_output = forward_propagation(X)
    error = backpropagation(X, Y, hidden_layer_output, final_output)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Error: {error}')

print("Final Output after Training:")
hidden_layer_input, hidden_layer_output, final_layer_input, final_output = forward_propagation(X)
print(final_output)




import numpy as np

def tanh(x):
    return np.tanh(x)
def forward_pass(x, w1, w2, b1, b2):
    # Hidden
    h = tanh(np.dot(w1, x) + b1)
    # Output
    y = tanh(np.dot(w2, h) + b2)
    return y
# weights in range [-0.5, 0.5]
n_input = 3   # input size
n_hidden = 4  # hidden layer size

np.random.seed(42)
w1 = np.random.uniform(-0.5, 0.5, (n_hidden, n_input))
w2 = np.random.uniform(-0.5, 0.5, n_hidden)

# biases
b1 = 0.5
b2 = 0.7

# input
x = np.array([0.2, -0.4, 0.6])
#  output
y_output = forward_pass(x, w1, w2, b1, b2)
print("Output of the network:", y_output)


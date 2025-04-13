import numpy as np

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): s = sigmoid(x); return s * (1 - s)

def tanh(x): return np.tanh(x)
def tanh_deriv(x): return 1 - np.tanh(x)**2

def leaky_relu(x): return np.where(x > 0, x, 0.01 * x)
def leaky_relu_deriv(x): return np.where(x > 0, 1, 0.01)

activation_functions = {
    "relu": (relu, relu_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
    "leaky_relu": (leaky_relu, leaky_relu_deriv)
}

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation="relu", reg=0.0):
        self.hidden_size = hidden_size
        self.reg = reg
        self.activation, self.activation_deriv = activation_functions[activation]

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        data_loss = np.sum(log_likelihood) / m
        reg_loss = 0.5 * self.reg * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return data_loss + reg_loss

    def backward(self, X, y_true):
        m = y_true.shape[0]
        y_onehot = np.zeros_like(self.a2)
        y_onehot[np.arange(m), y_true] = 1

        dz2 = (self.a2 - y_onehot) / m
        dW2 = self.a1.T @ dz2 + self.reg * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.activation_deriv(self.z1)
        dW1 = X.T @ dz1 + self.reg * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, lr):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        data = np.load(path)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']
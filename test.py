from model import NeuralNetwork
from utils import load_cifar10, preprocess_data, accuracy, visualize_weights

def test():
    _, _, _, _, X_test, y_test, input_dim, output_dim = preprocess_data()
    model = NeuralNetwork(input_dim, hidden_size=128, output_size=output_dim)
    model.load("best_model.npz")
    y_pred = model.forward(X_test)
    acc = accuracy(y_pred, y_test)
    print(f"Test Accuracy: {acc:.4f}")

def visualize_weights_func():
    X_train, y_train, X_val, y_val, X_test, y_test, input_dim, output_dim = preprocess_data()
    model = NeuralNetwork(input_size=input_dim, hidden_size=128, output_size=output_dim, activation="relu", reg=1e-3)
    model.load("best_model.npz")
    visualize_weights(model)

if __name__ == "__main__":
    test()
    visualize_weights_func()
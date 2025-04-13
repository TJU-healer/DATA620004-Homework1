import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        X = dict[b'data'].reshape(-1, 3, 32, 32).astype("float32") / 255.
        y = np.array(dict[b'labels'])
        X = X.transpose(0, 2, 3, 1).reshape(len(X), -1)
        return X, y

def load_cifar10(path='cifar-10-batches-py'):
    X_list, y_list = [], []
    for i in range(1, 6):
        X, y = load_cifar10_batch(os.path.join(path, f"data_batch_{i}"))
        X_list.append(X)
        y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)
    X_test, y_test = load_cifar10_batch(os.path.join(path, "test_batch"))
    return X_train, y_train, X_test, y_test

def preprocess_data():
    X_train, y_train, X_test, y_test = load_cifar10()
    X_val, y_val = X_train[45000:], y_train[45000:]
    X_train, y_train = X_train[:45000], y_train[:45000]
    return X_train, y_train, X_val, y_val, X_test, y_test, X_train.shape[1], 10

def accuracy(pred_probs, labels):
    preds = np.argmax(pred_probs, axis=1)
    return np.mean(preds == labels)

def visualize_weights(model, input_shape=(32, 32, 3), max_display=10):
    W1 = model.W1  # shape: (3072, hidden_size)
    num_units = min(W1.shape[1], max_display)
    plt.figure(figsize=(num_units * 2, 2))
    for i in range(num_units):
        w_img = W1[:, i].reshape(input_shape)
        w_img = (w_img - w_img.min()) / (w_img.max() - w_img.min())  # normalize
        plt.subplot(1, num_units, i + 1)
        plt.imshow(w_img)
        plt.axis('off')
        plt.title(f'Unit {i}')
    plt.tight_layout()
    plt.show()

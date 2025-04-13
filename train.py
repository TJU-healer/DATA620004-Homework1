from model import NeuralNetwork
from utils import load_cifar10, preprocess_data, accuracy
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product
import argparse

def train(X_train, y_train, X_val, y_val, input_dim, output_dim,
          hidden_size=128, lr=0.1, lr_decay=0.95, reg=1e-3,
          activation="relu", epochs=30, batch_size=128, 
          save_path="best_model.npz", verbose=True, plot=True):

    model = NeuralNetwork(input_dim, hidden_size, output_dim, activation=activation, reg=reg)
    best_acc = 0

    # Record metrics
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            y_pred = model.forward(X_batch)
            dW1, db1, dW2, db2 = model.backward(X_batch, y_batch)
            model.update_params(dW1, db1, dW2, db2, lr)

        lr *= lr_decay

        val_pred = model.forward(X_val)
        val_acc = accuracy(val_pred, y_val)
        val_loss = model.compute_loss(val_pred, y_val)

        train_pred = model.forward(X_train)
        train_loss = model.compute_loss(train_pred, y_train)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if verbose:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save(save_path)

    if plot:
        plot_metrics(train_losses, val_losses, val_accuracies)

    return best_acc


def plot_metrics(train_losses, val_losses, val_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def grid_search(X_train, y_train, X_val, y_val, input_dim, output_dim):
    learning_rates = [1e-1, 1e-2, 1e-3]
    hidden_sizes = [64, 128, 256]
    regularizations = [0.0, 1e-3, 1e-2]
    activations = ['relu', 'sigmoid', 'tanh']

    combinations = list(product(learning_rates, hidden_sizes, regularizations, activations))
    results = []

    print(f"\nGrid Search: Total {len(combinations)} combinations to evaluate...\n")

    for i, (lr, hidden, reg, act) in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Training with lr={lr}, hidden={hidden}, reg={reg}, activation={act}")

        model_name = f"model_lr{lr}_h{hidden}_reg{reg}_{act}.npz"
        acc = train(
            X_train, y_train, X_val, y_val,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden,
            lr=lr,
            lr_decay=0.95,
            reg=reg,
            activation=act,
            epochs=20,
            batch_size=128,
            save_path=os.path.join("checkpoints", model_name),
            verbose=False,
            plot=False
        )
        results.append((lr, hidden, reg, act, acc))
        print(f"Validation Acc: {acc:.4f}")

    # 排序展示最好的结果
    results.sort(key=lambda x: x[-1], reverse=True)
    print("\nTop 5 Best Configurations:")
    for r in results[:5]:
        print(f"lr={r[0]}, hidden={r[1]}, reg={r[2]}, activation={r[3]} --> Val Acc={r[4]:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--reg", type=float, default=1e-3, help="L2 regularization strength")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"], help="Activation function")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--save_path", type=str, default="best_model.npz", help="Path to save best model")
    parser.add_argument("--plot", action="store_true", help="Plot training curves")
    parser.add_argument("--verbose", action="store_true", help="Print training logs")
    parser.add_argument("--grid", action="store_true", help="Enable grid search mode")

    args = parser.parse_args()

    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, y_test, input_dim, output_dim = preprocess_data()

    if args.grid:
        os.makedirs("checkpoints", exist_ok=True)
        grid_search(X_train, y_train, X_val, y_val, input_dim, output_dim)
    else:
        train(
            X_train, y_train,
            X_val, y_val,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=args.hidden_size,
            lr=args.lr,
            lr_decay=0.95,
            reg=args.reg,
            activation=args.activation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.save_path,
            verbose=args.verbose,
            plot=args.plot
        )

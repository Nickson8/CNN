from lstm import LSTM
from data import plot_eeg, load_arff
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt



def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    plt.switch_backend('TkAgg')
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    #DATA
    plot_eeg()

    df = load_arff('filtered_output.arff')

    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy(dtype=int)

    bs = 8

    #One-Hot Enconding y
    Y = (np.eye(2)[y])[:-(y.shape[0] % bs)]


    #Standardization of X
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_standardized = ((X - mean) / std)[:-(X.shape[0] % bs)]

    print(X.shape[0])
    print(Y.shape[0])



    # Load and preprocess MNIST data
    x_train, y_train, x_test, y_test = (X_standardized, Y, X_standardized, Y)

    # Use a smaller subset for faster training and demonstration
    train_samples = X_standardized.shape[0]  # Adjust this number as needed
    test_samples = 300  # Adjust this number as needed

    x_train_subset = x_train[:train_samples]
    y_train_subset = y_train[:train_samples]
    x_test_subset = x_test[:test_samples]
    y_test_subset = y_test[:test_samples]

    # Initialize CNN model
    lstm = LSTM(batch_size=bs, hidden_units=16, lr=0.01)

    lstm.fit(X=x_train_subset, Y=y_train_subset, validation_data=(x_test_subset, y_test_subset))

    # Train the model
    print("Training the model...")
    history = lstm.train(epochs=20)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_loss = lstm.evaluate(x_test_subset, y_test_subset)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Plot training history
    lstm.plot_history(history)


if __name__ == "__main__":
    main()
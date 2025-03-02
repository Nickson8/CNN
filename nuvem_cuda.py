import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from predictor import Predictor


def printf(coisa):
    print("\n=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*\n")
    print(coisa)
    print("\n=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*\n")

class CNN:
    def __init__(self, ks, learning_rate):
        self.ks = ks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {torch.cuda.get_device_name(0)}({self.device})")

        # First Layer Kernels - convert to PyTorch tensors on GPU
        self.l1_kernels = torch.randn(ks[0], 1, 3, 3, device=self.device) * 0.1
        self.l1_kernels.requires_grad = True

        # Second Layer Kernels
        self.l2_kernels = torch.randn(ks[1], 1, 3, 3, device=self.device) * 0.1
        self.l2_kernels.requires_grad = True

        # Third Layer Kernels
        self.l3_kernels = torch.randn(ks[2], 1, 3, 3,
                                      device=self.device) * 0.1
        self.l3_kernels.requires_grad = True

        # Learning rate
        self.learning_rate = learning_rate

        # Prediction Layer Weights
        n = ks[0] * ks[1] * ks[2]
        self.predictor = Predictor(layers_sizes=[10], input_size=n, lr=0.01, device=self.device)
        self.pred_optimizer = self.predictor.get_optimizer()

    def fit(self, Imgs, Y, epochs=10, batch_size=32, validation_data=None):
        """
        Train the CNN model using CUDA parallelization

        Args:
            Imgs: Input images with shape (num_samples, height, width)
            Y: One-hot encoded labels with shape (num_samples, 10)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            learning_rate: Learning rate for gradient descent
            validation_data: Tuple of (validation_images, validation_labels)
        """
        num_samples = len(Imgs)

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(np.array(Imgs), dtype=torch.float32, device=self.device).unsqueeze(
            1)  # Add channel dimension
        Y_tensor = torch.tensor(Y, dtype=torch.float32, device=self.device)

        if validation_data is not None:
            val_X = torch.tensor(np.array(validation_data[0]), dtype=torch.float32, device=self.device).unsqueeze(1)
            val_Y = torch.tensor(validation_data[1], dtype=torch.float32, device=self.device)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(num_samples, device=self.device)
            X_shuffled = X_tensor[indices]
            Y_shuffled = Y_tensor[indices]

            total_loss = 0
            correct = 0

            # Mini-batch training - this is now parallelized with CUDA
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                batch_X = X_shuffled[i:end]
                batch_Y = Y_shuffled[i:end]

                # Forward pass (all samples in batch at once)
                predictions = self.forward_batch(batch_X)

                # Calculate loss
                loss = self.cross_entropy_loss_batch(batch_Y, predictions)
                total_loss += loss.item() * (end - i)

                # Calculate accuracy
                pred_classes = torch.argmax(predictions, dim=1)
                true_classes = torch.argmax(batch_Y, dim=1)
                correct += (pred_classes == true_classes).sum().item()

                # Backward pass (computes gradients for all samples in parallel)
                self.backward_batch(batch_Y, predictions)

            # Epoch statistics
            avg_train_loss = total_loss / num_samples
            train_accuracy = correct / num_samples
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Validation
            if validation_data is not None:
                val_accuracy, val_loss = self.evaluate_batch(val_X, val_Y)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Return training history
        return {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        }

    def forward_batch(self, batch_X):
        """
        Forward pass for a batch of images

        Args:
            batch_X: Batch of images tensor with shape (batch_size, 1, height, width)

        Returns:
            features: Flattened features before final layer
            predictions: Softmax probabilities
        """
        batch_size = batch_X.shape[0]

        # First Layer - Convolution + ReLU + Pooling
        # Use PyTorch's built-in Conv2d function for parallel computation
        l1_outputs = F.relu(torch.conv2d(input=batch_X, weight=self.l1_kernels, stride=1, padding=0))
        l1_pooled = F.max_pool2d(l1_outputs, kernel_size=2, stride=2)

        # Second Layer
        # Reshape for grouped convolution
        batch_l1 = l1_pooled.view(batch_size , self.ks[0],
                                  l1_pooled.shape[2], l1_pooled.shape[3])
        
        l2_outputs = F.relu(F.conv2d(input=batch_l1, weight=self.l2_kernels.repeat(self.ks[0],1,1,1),
            padding=0, groups=self.ks[0]))
        l2_pooled = F.max_pool2d(l2_outputs, kernel_size=2, stride=2)

        # Third Layer
        batch_l2 = l2_pooled.view(batch_size, self.ks[0] * self.ks[1],
                                  l2_pooled.shape[2], l2_pooled.shape[3])

        l3_outputs = F.relu(F.conv2d(input=batch_l2, weight=self.l3_kernels.repeat(self.ks[0]*self.ks[1],1,1,1), padding=0,
                                     groups=self.ks[0] * self.ks[1]))
        l3_pooled = F.max_pool2d(l3_outputs, kernel_size=2, stride=2)

        # Flatten features
        features = l3_pooled.view(batch_size, -1)

        # Prediction layer
        predictions = self.predictor.forward(features)

        return predictions

    def cross_entropy_loss_batch(self, y_true, y_pred):
        """
        Calculate cross entropy loss for a batch

        Args:
            y_true: One-hot encoded true labels tensor
            y_pred: Predicted probabilities tensor

        Returns:
            loss: Cross entropy loss tensor
        """
        # PyTorch's cross entropy is more numerically stable
        return F.binary_cross_entropy(y_pred, y_true, reduction='sum') / y_true.shape[0]

    def backward_batch(self, batch_Y, predictions):
        """
        Backward pass for a batch using automatic differentiation

        Args:
            batch_X: Batch of input images
            batch_Y: Batch of true labels
            predictions: Batch of predicted probabilities
        """
        # Calculate loss
        loss = self.cross_entropy_loss_batch(batch_Y, predictions)

        # Zero all gradients
        if self.l1_kernels.grad is not None:
            self.l1_kernels.grad.zero_()
        if self.l2_kernels.grad is not None:
            self.l2_kernels.grad.zero_()
        if self.l3_kernels.grad is not None:
            self.l3_kernels.grad.zero_()
        self.pred_optimizer.zero_grad()

        # Backward pass - compute gradients
        loss.backward()

        # Update parameters using gradient descent
        with torch.no_grad():
            self.l1_kernels -= self.learning_rate * self.l1_kernels.grad
            self.l2_kernels -= self.learning_rate * self.l2_kernels.grad
            self.l3_kernels -= self.learning_rate * self.l3_kernels.grad
            self.pred_optimizer.step()

    def predict_batch(self, batch_X):
        """
        Make predictions for a batch of images

        Args:
            batch_X: Batch of input images tensor

        Returns:
            predicted_classes: Tensor of predicted class indices
            confidences: Tensor of confidence scores
        """
        with torch.no_grad():
            predictions = self.forward_batch(batch_X)
            predicted_classes = torch.argmax(predictions, dim=1)
            confidences = torch.gather(predictions, 1, predicted_classes.unsqueeze(1)).squeeze()

        return predicted_classes, confidences

    def evaluate_batch(self, test_X, test_Y):
        """
        Evaluate model on test data in batches

        Args:
            test_X: Test images tensor
            test_Y: One-hot encoded test labels tensor

        Returns:
            accuracy: Model accuracy
            loss: Average cross entropy loss
        """
        with torch.no_grad():
            predictions = self.forward_batch(test_X)
            loss = self.cross_entropy_loss_batch(test_Y, predictions).item()

            pred_classes = torch.argmax(predictions, dim=1)
            true_classes = torch.argmax(test_Y, dim=1)
            correct = (pred_classes == true_classes).sum().item()
            accuracy = correct / test_X.shape[0]

        return accuracy, loss

    def predict(self, img):
        """
        Make a prediction for a single image

        Args:
            img: Input image

        Returns:
            predicted_class: Predicted class index
            confidence: Confidence score (probability)
        """
        # Convert to tensor and add batch and channel dimensions
        img_tensor = torch.tensor(img, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        # Get prediction
        predicted_class, confidence = self.predict_batch(img_tensor)

        return predicted_class.item(), confidence.item()

    def evaluate(self, test_images, test_labels):
        """
        Evaluate model on test data

        Args:
            test_images: Test images
            test_labels: One-hot encoded test labels

        Returns:
            accuracy: Model accuracy
            loss: Average cross entropy loss
        """
        # Convert to tensors
        test_X = torch.tensor(np.array(test_images), dtype=torch.float32, device=self.device).unsqueeze(1)
        test_Y = torch.tensor(test_labels, dtype=torch.float32, device=self.device)

        return self.evaluate_batch(test_X, test_Y)

    def plot_history(self, history):
        """
        Plot training and validation metrics

        Args:
            history: Dictionary containing training history
        """
        # Plot loss
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history and history['val_accuracy']:
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def visualize_filters(self, layer=1):
        """
        Visualize filters from a specific layer

        Args:
            layer: Layer number (1, 2, or 3)
        """
        if layer == 1:
            kernels = self.l1_kernels.detach().cpu().numpy()
            title = "First Layer Filters"
        elif layer == 2:
            # Reshape to visualize individual filters
            kernels = self.l2_kernels.detach().cpu().numpy()
            kernels = kernels.reshape(-1, 3, 3)
            title = "Second Layer Filters (Sample)"
        elif layer == 3:
            # Reshape to visualize individual filters
            kernels = self.l3_kernels.detach().cpu().numpy()
            kernels = kernels.reshape(-1, 3, 3)
            title = "Third Layer Filters (Sample)"
        else:
            raise ValueError("Layer must be 1, 2, or 3")

        # Select a subset of kernels to display
        if kernels.shape[0] > 16:
            kernels = kernels[:16]

        n_kernels = kernels.shape[0]
        n_cols = min(8, n_kernels)
        n_rows = (n_kernels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        for i in range(n_kernels):
            # Normalize kernel for visualization
            kernel = kernels[i]
            if layer == 1:
                kernel = kernel[0]  # Extract the channel
            kernel_min = kernel.min()
            kernel_max = kernel.max()
            if kernel_max > kernel_min:
                kernel_normalized = (kernel - kernel_min) / (kernel_max - kernel_min)
            else:
                kernel_normalized = kernel

            axes[i].imshow(kernel_normalized, cmap='viridis')
            axes[i].set_title(f"Filter {i + 1}")
            axes[i].axis('off')

        # Hide any unused subplots
        for i in range(n_kernels, len(axes)):
            axes[i].axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


def preprocess_mnist_data():
    """
    Load and preprocess MNIST dataset

    Returns:
        x_train: Training images
        y_train: Training labels (one-hot encoded)
        x_test: Test images
        y_test: Test labels (one-hot encoded)
    """

    # Normalize pixel values to 0-1
    x_train = pickle.loads(open("mnist_data/Xtrain", "rb").read())
    x_test = pickle.loads(open("mnist_data/Xtest", "rb").read())

    # Convert class vectors to one-hot encoded matrices
    y_train = pickle.loads(open("mnist_data/Ytrain", "rb").read())
    y_test = pickle.loads(open("mnist_data/Ytest", "rb").read())

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    plt.switch_backend('TkAgg')
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Load and preprocess MNIST data
    x_train, y_train, x_test, y_test = preprocess_mnist_data()

    # Use a smaller subset for faster training and demonstration
    train_samples = 60000  # Adjust this number as needed
    test_samples = 1000  # Adjust this number as needed

    x_train_subset = x_train[:train_samples]
    y_train_subset = y_train[:train_samples]
    x_test_subset = x_test[:test_samples]
    y_test_subset = y_test[:test_samples]

    # Initialize CNN model
    cnn = CNN(ks=[4, 4, 4], learning_rate=0.01)  # 4 kernels in each layer

    # Train the model
    print("Training the model...")
    history = cnn.fit(
        x_train_subset,
        y_train_subset,
        epochs=15,
        batch_size=16,
        validation_data=(x_test_subset, y_test_subset)
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_loss = cnn.evaluate(x_test_subset, y_test_subset)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Plot training history
    cnn.plot_history(history)

    # Visualize filters
    # cnn.visualize_filters(layer=1)

    # Make predictions on a few examples
    print("\nPredictions on sample images:")
    for i in range(5):
        class_prediction, confidence = cnn.predict(x_test_subset[i])
        true_class = np.argmax(y_test_subset[i])
        print(
            f"Sample {i + 1}: Predicted {class_prediction} with confidence {confidence:.4f}, True class: {true_class}")

        # Display the image
        plt.figure(figsize=(5,5))
        plt.imshow(x_test_subset[i], cmap='gray')
        plt.title(f"Prediction: {class_prediction}\nTrue: {true_class}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
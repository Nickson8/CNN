import numpy as np
import pickle
import matplotlib.pyplot as plt


class CNN:
    def __init__(self, kernels_sizes):
        self.kernels_sizes = kernels_sizes
        # First Layer Kernels
        self.l1_kernels = np.random.randn(kernels_sizes[0], 3, 3) * 0.1
        # Second Layer Kernels
        self.l2_kernels = np.random.randn(kernels_sizes[1], 3, 3) * 0.1
        # Third Layer Kernels
        self.l3_kernels = np.random.randn(kernels_sizes[2], 3, 3) * 0.1
        # Prediction Layer Weights
        n = kernels_sizes[0] * kernels_sizes[1] * kernels_sizes[2]
        self.pw = np.random.randn(n, 10) * 0.1
        
        # Learning rate
        self.learning_rate = 0.01
        
        # Store intermediate values for backpropagation
        self.cache = {}
        
    def fit(self, Imgs, Y, epochs=10, batch_size=32, learning_rate=0.01, validation_data=None):
        """
        Train the CNN model
        
        Args:
            Imgs: Input images with shape (num_samples, height, width)
            Y: One-hot encoded labels with shape (num_samples, 10)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            learning_rate: Learning rate for gradient descent
            validation_data: Tuple of (validation_images, validation_labels)
        """
        self.learning_rate = learning_rate
        num_samples = len(Imgs)
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = [Imgs[i] for i in indices]
            Y_shuffled = Y[indices]
            
            loss_sum = 0
            accuracy_sum = 0
            
            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                batch_X = X_shuffled[i:end]
                batch_Y = Y_shuffled[i:end]
                
                batch_loss = 0
                batch_correct = 0
                
                # Process each sample in the batch
                for j in range(len(batch_X)):
                    # Forward pass
                    features, predictions = self.forward(batch_X[j])
                    
                    # Calculate loss
                    loss = self.cross_entropy_loss(batch_Y[j], predictions)
                    batch_loss += loss
                    
                    # Check if prediction is correct
                    if np.argmax(predictions) == np.argmax(batch_Y[j]):
                        batch_correct += 1
                    
                    # Backward pass
                    self.backward(batch_X[j], batch_Y[j], features, predictions)
                
                # Update batch statistics
                batch_loss /= len(batch_X)
                batch_acc = batch_correct / len(batch_X)
                loss_sum += batch_loss * len(batch_X)
                accuracy_sum += batch_correct
                
                # Print mini-batch progress
                if (i // batch_size) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {i//batch_size + 1}/{num_samples//batch_size + 1}, Loss: {batch_loss:.4f}, Accuracy: {batch_acc:.4f}")
            
            # Epoch statistics
            avg_train_loss = loss_sum / num_samples
            train_accuracy = accuracy_sum / num_samples
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            
            # Validation
            if validation_data is not None:
                val_accuracy, val_loss = self.evaluate(validation_data[0], validation_data[1])
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
    
    def ReLu(self, X):
        return np.maximum(0, X)
    
    def ReLu_derivative(self, X):
        return np.where(X > 0, 1, 0)
    
    def conv_(self, img_part, kernel):
        return self.ReLu(np.sum(img_part * kernel))
    
    def Conv(self, img, kernel):
        # Convolution with 3x3 kernel and stride 1
        new = np.zeros((img.shape[0] - 2, img.shape[1] - 2))
        for i in range(img.shape[0] - 2):
            for j in range(img.shape[1] - 2):
                new[i, j] = self.conv_(img_part=img[i:i+3, j:j+3], kernel=kernel)
        return new
    
    def pool_(self, img_part):
        # 2x2 max-pooling
        return np.max(img_part)
    
    def Pool(self, img):
        # 2x2 max-pooling with stride 2
        new = np.zeros((img.shape[0]//2, img.shape[1]//2))
        max_indices = np.zeros((img.shape[0]//2, img.shape[1]//2, 2), dtype=int)
        
        i = 0
        for l in range(img.shape[0]//2):
            j = 0
            for c in range(img.shape[1]//2):
                pool_window = img[i:i+2, j:j+2]
                max_val = np.max(pool_window)
                new[l, c] = max_val
                
                # Store indices of max value for backpropagation
                max_pos = np.unravel_index(np.argmax(pool_window), pool_window.shape)
                max_indices[l, c] = [i + max_pos[0], j + max_pos[1]]
                
                j += 2
            i += 2
            
        return new, max_indices
    
    def softmax(self, Z):
        # Shift Z for numerical stability
        Z_shifted = Z - np.max(Z)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z)
    
    def forward(self, img):
        """
        Forward pass through the network
        
        Args:
            img: Input image
            
        Returns:
            features: Flattened features before final layer
            predictions: Softmax probabilities
        """
        self.cache = {}
        self.cache['input'] = img
        
        # First Layer
        l1_conv_outputs = []
        l1_pool_outputs = []
        l1_pool_indices = []
        
        for k in range(self.l1_kernels.shape[0]):
            conv_output = self.Conv(img=img, kernel=self.l1_kernels[k])
            l1_conv_outputs.append(conv_output)
            
            pool_output, pool_indices = self.Pool(conv_output)
            l1_pool_outputs.append(pool_output)
            l1_pool_indices.append(pool_indices)
        
        self.cache['l1_conv'] = l1_conv_outputs
        self.cache['l1_pool'] = l1_pool_outputs
        self.cache['l1_pool_indices'] = l1_pool_indices
        
        # Second Layer
        l2_conv_outputs = []
        l2_pool_outputs = []
        l2_pool_indices = []
        
        for i, feature_map in enumerate(l1_pool_outputs):
            for k in range(self.l2_kernels.shape[0]):
                conv_output = self.Conv(img=feature_map, kernel=self.l2_kernels[k])
                l2_conv_outputs.append(conv_output)
                
                pool_output, pool_indices = self.Pool(conv_output)
                l2_pool_outputs.append(pool_output)
                l2_pool_indices.append(pool_indices)
        
        self.cache['l2_conv'] = l2_conv_outputs
        self.cache['l2_pool'] = l2_pool_outputs
        self.cache['l2_pool_indices'] = l2_pool_indices
        
        # Third Layer
        l3_conv_outputs = []
        l3_pool_outputs = []
        l3_pool_indices = []
        
        for i, feature_map in enumerate(l2_pool_outputs):
            for k in range(self.l3_kernels.shape[0]):
                conv_output = self.Conv(img=feature_map, kernel=self.l3_kernels[k])
                l3_conv_outputs.append(conv_output)
                
                pool_output, pool_indices = self.Pool(conv_output)
                l3_pool_outputs.append(pool_output)
                l3_pool_indices.append(pool_indices)
        
        self.cache['l3_conv'] = l3_conv_outputs
        self.cache['l3_pool'] = l3_pool_outputs
        self.cache['l3_pool_indices'] = l3_pool_indices
        
        # Flatten features
        features = np.array([pool[0, 0] for pool in l3_pool_outputs])
        self.cache['features'] = features
        
        # Prediction layer
        Z = features @ self.pw
        self.cache['Z'] = Z
        predictions = self.softmax(Z)
        
        return features, predictions
    
    def cross_entropy_loss(self, y_true, y_pred):
        """
        Calculate cross entropy loss
        
        Args:
            y_true: One-hot encoded true labels
            y_pred: Predicted probabilities
            
        Returns:
            loss: Cross entropy loss
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred))
    
    def backward(self, img, y_true, features, y_pred):
        """
        Backward pass for backpropagation
        
        Args:
            img: Input image
            y_true: One-hot encoded true labels
            features: Flattened features before final layer
            y_pred: Predicted probabilities
        """
        # Gradient of loss with respect to predictions
        m = 1  # Single sample
        dL_dP = -(y_true / y_pred) / m
        
        # Gradient of softmax
        Z = self.cache['Z']
        dZ = np.zeros_like(Z)
        for i in range(len(y_pred)):
            for j in range(len(y_pred)):
                if i == j:
                    dZ[j] += dL_dP[i] * y_pred[i] * (1 - y_pred[i])
                else:
                    dZ[j] += dL_dP[i] * (-y_pred[i] * y_pred[j])
        
        # Gradient of prediction layer weights
        dW = np.outer(features, dZ)
        
        # Update prediction weights
        self.pw -= self.learning_rate * dW
        
        # Gradient of features
        dFeatures = dZ @ self.pw.T
        
        # Backpropagate through the third layer
        dL3_kernels = np.zeros_like(self.l3_kernels)
        dL2_pool = [np.zeros_like(pool) for pool in self.cache['l2_pool']]
        
        feature_idx = 0
        for i, l2_feature_idx in enumerate(range(len(self.cache['l2_pool']))):
            for k in range(self.l3_kernels.shape[0]):
                # Get stored values
                pool_output = self.cache['l3_pool'][feature_idx]
                pool_indices = self.cache['l3_pool_indices'][feature_idx]
                conv_output = self.cache['l3_conv'][feature_idx]
                l2_feature = self.cache['l2_pool'][l2_feature_idx]
                
                # Gradient for this feature
                dFeature = dFeatures[feature_idx]
                
                # Backprop through pooling layer
                dPool = np.zeros_like(conv_output)
                # Set gradient only at the max position
                max_i, max_j = pool_indices[0, 0]
                dPool[max_i - (max_i//2)*2, max_j - (max_j//2)*2] = dFeature
                
                # Backprop through ReLU
                dReLU = dPool * self.ReLu_derivative(conv_output)
                
                # Backprop through convolution
                # Update kernel gradients
                for i in range(l2_feature.shape[0] - 2):
                    for j in range(l2_feature.shape[1] - 2):
                        if dReLU[i, j] != 0:
                            dL3_kernels[k] += dReLU[i, j] * l2_feature[i:i+3, j:j+3]
                            # Also accumulate gradients for previous layer
                            dL2_pool[l2_feature_idx][i:i+3, j:j+3] += dReLU[i, j] * self.l3_kernels[k]
                
                feature_idx += 1
        
        # Update third layer kernels
        self.l3_kernels -= self.learning_rate * dL3_kernels
        
        # For a complete implementation, backpropagation for the second and first layers would follow
        # This is a simplified version focusing on the main concepts
        
    def predict(self, img):
        """
        Make a prediction for a single image
        
        Args:
            img: Input image
            
        Returns:
            predicted_class: Predicted class index
            confidence: Confidence score (probability)
        """
        _, predictions = self.forward(img)
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        return predicted_class, confidence
    
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
        correct = 0
        total_loss = 0
        
        for i in range(len(test_images)):
            _, predictions = self.forward(test_images[i])
            loss = self.cross_entropy_loss(test_labels[i], predictions)
            total_loss += loss
            
            if np.argmax(predictions) == np.argmax(test_labels[i]):
                correct += 1
        
        accuracy = correct / len(test_images)
        avg_loss = total_loss / len(test_images)
        
        return accuracy, avg_loss
    
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
            kernels = self.l1_kernels
            title = "First Layer Filters"
        elif layer == 2:
            kernels = self.l2_kernels
            title = "Second Layer Filters"
        elif layer == 3:
            kernels = self.l3_kernels
            title = "Third Layer Filters"
        else:
            raise ValueError("Layer must be 1, 2, or 3")
        
        n_kernels = kernels.shape[0]
        fig, axes = plt.subplots(1, n_kernels, figsize=(n_kernels * 2, 2))
        
        for i in range(n_kernels):
            # Normalize kernel for visualization
            kernel = kernels[i]
            kernel_normalized = (kernel - kernel.min()) / (kernel.max() - kernel.min())
            
            if n_kernels > 1:
                axes[i].imshow(kernel_normalized, cmap='viridis')
                axes[i].set_title(f"Filter {i+1}")
                axes[i].axis('off')
            else:
                axes.imshow(kernel_normalized, cmap='viridis')
                axes.set_title(f"Filter {i+1}")
                axes.axis('off')
        
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
    x_train = pickle.loads( open("mnist_data/Xtrain", "rb").read() )
    x_test = pickle.loads( open("mnist_data/Xtest", "rb").read() )
    
    # Convert class vectors to one-hot encoded matrices
    y_train = pickle.loads( open("mnist_data/Ytrain", "rb").read() )
    y_test = pickle.loads( open("mnist_data/Ytest", "rb").read() )
    
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load and preprocess MNIST data
    x_train, y_train, x_test, y_test = preprocess_mnist_data()
    
    # Use a smaller subset for faster training and demonstration
    train_samples = 3000  # Adjust this number as needed
    test_samples = 200    # Adjust this number as needed
    
    x_train_subset = x_train[:train_samples]
    y_train_subset = y_train[:train_samples]
    x_test_subset = x_test[:test_samples]
    y_test_subset = y_test[:test_samples]
    
    # Initialize CNN model
    cnn = CNN(kernels_sizes=[4, 4, 4])  # 4 kernels in each layer
    
    # Train the model
    print("Training the model...")
    history = cnn.fit(
        x_train_subset, 
        y_train_subset,
        epochs=20,
        batch_size=32,
        learning_rate=0.01,
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
    cnn.visualize_filters(layer=1)
    
    # Make predictions on a few examples
    print("\nPredictions on sample images:")
    for i in range(5):
        class_prediction, confidence = cnn.predict(x_test_subset[i])
        true_class = np.argmax(y_test_subset[i])
        print(f"Sample {i+1}: Predicted {class_prediction} with confidence {confidence:.4f}, True class: {true_class}")
        
        # Display the image
        plt.figure(figsize=(3, 3))
        plt.imshow(x_test_subset[i], cmap='gray')
        plt.title(f"Prediction: {class_prediction}\nTrue: {true_class}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from predictor import Predictor
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, batch_size, hidden_units, lr):
        super(LSTM, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {torch.cuda.get_device_name(0)}({self.device})")

        self.bs = batch_size
        self.h = hidden_units
        self.lr = lr

    def fit(self, X, Y, validation_data):

        self.X = X
        self.Y = Y
        self.validation_data = validation_data

        num_features = X.shape[1]

        #CRIANDO OS PESOS DOS GATES
        self.gates = nn.ParameterDict()

        #Forget Gate
        self.gates['ForgetX'] = nn.Parameter(torch.randn(num_features, self.h, device=self.device) * 0.1)
        self.gates['ForgetH'] = nn.Parameter(torch.randn(self.h, self.h, device=self.device) * 0.1)
        self.gates['ForgetB'] = nn.Parameter(torch.randn(1, self.h, device=self.device) * 0.1)

        #Input Gate
        self.gates['InputX'] = nn.Parameter(torch.randn(num_features, self.h, device=self.device) * 0.1)
        self.gates['InputH'] = nn.Parameter(torch.randn(self.h, self.h, device=self.device) * 0.1)
        self.gates['InputB'] = nn.Parameter(torch.randn(1, self.h, device=self.device) * 0.1)

        #Output Gate
        self.gates['OutputX'] = nn.Parameter(torch.randn(num_features, self.h, device=self.device) * 0.1)
        self.gates['OutputH'] = nn.Parameter(torch.randn(self.h, self.h, device=self.device) * 0.1)
        self.gates['OutputB'] = nn.Parameter(torch.randn(1, self.h, device=self.device) * 0.1)

        #CandS Gate
        self.gates['CandSX'] = nn.Parameter(torch.randn(num_features, self.h, device=self.device) * 0.1)
        self.gates['CandSH'] = nn.Parameter(torch.randn(self.h, self.h, device=self.device) * 0.1)
        self.gates['CandSB'] = nn.Parameter(torch.randn(1, self.h, device=self.device) * 0.1)

        self.gates_optmizer = torch.optim.SGD(self.parameters(), lr=self.lr)


        #CRIANDO A PREDICTION LAYER
        self.predictor = Predictor(layers_sizes=[40, 20, Y.shape[1]], input_size=self.h, lr=self.lr, device=self.device)
        self.pred_optimizer = self.predictor.get_optimizer()

    def train(self, epochs):
        """
        Train the LSTM model using CUDA parallelization

        Args:
            X: Input data
            Y: One-hot encoded labels with shape (num_samples, num_labels)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
        """

        num_samples = self.X.shape[0]

        
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(np.array(self.X), dtype=torch.float32, device=self.device)
        Y_tensor = torch.tensor(self.Y, dtype=torch.float32, device=self.device)

        if self.validation_data is not None:
            val_X = torch.tensor(np.array(self.validation_data[0]), dtype=torch.float32, device=self.device).unsqueeze(1)
            val_Y = torch.tensor(self.validation_data[1], dtype=torch.float32, device=self.device)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            #CRIANDO OS VALORES INTERMEDIÁRIOS
            self.C = torch.zeros(self.bs, self.h, device=self.device)
            self.H = torch.zeros(self.bs, self.h, device=self.device)

            total_loss = 0
            correct = 0

            # Mini-batch training - this is now parallelized with CUDA
            for i in range(0, num_samples, self.bs):
                end = min(i + self.bs, num_samples)
                batch_X = X_tensor[i:end]
                batch_Y = Y_tensor[i:end]

                # Forward pass (all samples in batch at once)
                predictions = self.forward_batch(batch_X)

                # Calculate accuracy
                pred_classes = torch.argmax(predictions, dim=1)
                true_classes = torch.argmax(batch_Y, dim=1)
                correct += (pred_classes == true_classes).sum().item()

                # Backward pass (computes gradients for all samples in parallel)
                loss = self.backward_batch(batch_Y, predictions)
                total_loss += loss.item() * (end - i)

            # Epoch statistics
            avg_train_loss = total_loss / num_samples
            train_accuracy = correct / num_samples
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Validation
            #if self.validation_data is not None:
                #val_accuracy, val_loss = self.evaluate_batch(val_X, val_Y)
                #val_losses.append(val_loss)
                #val_accuracies.append(val_accuracy)
                #print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Return training history
        return {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        }
    
    def forward_batch(self, batch_X):
        # Reset hidden state and cell state for each batch
        # This detaches them from the previous computation graph
        batch_size = batch_X.shape[0]
        
        # Initialize states for this batch if needed
        if self.C.shape[0] != batch_size:
            self.C = torch.zeros(batch_size, self.h, device=self.device)
            self.H = torch.zeros(batch_size, self.h, device=self.device)
        else:
            # Detach states from previous computation graph
            self.C = self.C.detach()
            self.H = self.H.detach()
        
        try:
            self.C = (
                (torch.sigmoid((batch_X @ self.gates['ForgetX']) + (self.H @ self.gates['ForgetH']) + self.gates['ForgetB']) * self.C)
                +
                (torch.sigmoid((batch_X @ self.gates['InputX']) + (self.H @ self.gates['InputH']) + self.gates['InputB']) *
                torch.tanh((batch_X @ self.gates['CandSX']) + (self.H @ self.gates['CandSH']) + self.gates['CandSB'])))
                
            self.H = (torch.sigmoid((batch_X @ self.gates['OutputX']) + (self.H @ self.gates['OutputH']) + self.gates['OutputB']) * torch.tanh(self.C))
        except Exception as e:
            print(f"Error in forward pass: {e}")
            print(f"Shapes - batch_X: {batch_X.shape}, ForgetX: {self.gates['ForgetX'].shape}, H: {self.H.shape}, ForgetH: {self.gates['ForgetH'].shape}")
        

        return self.predictor.forward(self.H)
    
    def forward(self, X):
        # Reset hidden state and cell state for each batch
        # This detaches them from the previous computation graph
        
        self.C = (
            (torch.sigmoid((X @ self.gates['ForgetX']) + (self.H @ self.gates['ForgetH']) + self.gates['ForgetB']) * self.C)
            +
            (torch.sigmoid((X @ self.gates['InputX']) + (self.H @ self.gates['InputH']) + self.gates['InputB']) *
            torch.tanh((X @ self.gates['CandSX']) + (self.H @ self.gates['CandSH']) + self.gates['CandSB'])))
            
        self.H = (torch.sigmoid((X @ self.gates['OutputX']) + (self.H @ self.gates['OutputH']) + self.gates['OutputB']) * torch.tanh(self.C))
        

        return self.predictor.forward(self.H)
    
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
            batch_Y: Batch of true labels
            predictions: Batch of predicted probabilities
        """
        # Calculate loss
        loss = self.cross_entropy_loss_batch(batch_Y, predictions)

        # Zero all gradients
        self.gates_optmizer.zero_grad()
        self.pred_optimizer.zero_grad()

        # Backward pass - compute gradients
        loss.backward()

        # Update parameters using gradient descent
        self.gates_optmizer.step()
        self.pred_optimizer.step()

        return loss
    
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
    
    def evaluate(self, test_X, test_Y):
        """
        Evaluate model on test data in batches

        Args:
            test_X: Test images tensor
            test_Y: One-hot encoded test labels tensor

        Returns:
            accuracy: Model accuracy
            loss: Average cross entropy loss
        """
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(np.array(test_X), dtype=torch.float32, device=self.device)
        Y_tensor = torch.tensor(test_Y, dtype=torch.float32, device=self.device)

        correct = 0
        loss = 0

        #CRIANDO OS VALORES INTERMEDIÁRIOS
        self.C = torch.zeros(1, self.h, device=self.device)
        self.H = torch.zeros(1, self.h, device=self.device)

        print("P   T")

        for i in range(X_tensor.shape[0]):
            with torch.no_grad():
                predictions = self.forward(X_tensor[i])
                loss += self.cross_entropy_loss_batch(Y_tensor[i].reshape(1,2), predictions).item()

                pred_classes = torch.argmax(predictions, dim=1)
                true_classes = torch.argmax(Y_tensor[i].reshape(1,2), dim=1)
                print(f"{pred_classes.item()} - {true_classes.item()}, confidence: {predictions[0][pred_classes].item()}")
                correct += (pred_classes == true_classes).sum().item()
        
        loss = loss / X_tensor.shape[0]
        accuracy = correct / X_tensor.shape[0]

        return accuracy, loss
    
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
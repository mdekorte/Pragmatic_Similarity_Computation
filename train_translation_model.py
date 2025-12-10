import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset


class MLP(torch.nn.Module):
    """A simple Multi-Layer Perceptron (MLP) model.
    Args:
        layers: A list of tuples where each tuple contains the number of input and output neurons for each layer.
        hidden_activation: The activation function to use for the hidden layers.
        output: A tuple containing the number of input and output neurons for the output layer.
    """
    def __init__(self, layers, hidden_activation, output):
        super(MLP,self).__init__()
        self.architecture = nn.Sequential()
        for neurons in layers:
            self.architecture.append(nn.Linear(neurons[0], neurons[1]))
            self.architecture.append(hidden_activation())
        self.architecture.append(nn.Linear(output[0], output[1]))


    def forward(self,x):
        return self.architecture(x)


def train_test_model(model, X_train, X_test, y_train, y_test, epochs=200, lr=0.01, batch_size=200):
    """
    Prepares data, trains the model, and returns evaluation results.

    Args:
        model: The PyTorch model to train and test.
        X_train: The input data that will be used to train the model.
        X_test: The input data that will be used to evaluate the model's performance.
        y_train: The expected output data that the model will use during training.
        y_test: The expected output data used to compare the model's performance during testing.
        epochs: The number of iterations that the entire training data will be seen.
        lr: The initial learning rate at which the optimizer is initialized
        batch_size: The number of instances contained in each batch when splitting the entire dataset.

    Returns:
        A tuple containing the MAE, MSE, and the Mean Cosine Similarity of the model with respect to the testing data.
    """

    # Convert into a np array instead of a list of np arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training the model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    train_model(model, optimizer, criterion, train_loader, epochs)

    # Testing the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)

    return compute_eval_metrics(y_pred, y_test_tensor)

def train_model(model, optimizer, criterion, train_loader, epochs=200):
    """
    Standard training loop.

    Args:
        model: The PyTorch model to train
        optimizer: The PyTorch optimizer used to update the model parameters
        criterion: The Loss PyTorch object used to calculate the gradients of the model
        train_loader: The util object that helps load the training data in batches
        epochs: The number of iterations that the entire training data will be seen

    """
    model.train()
    # Training loop
    for epoch in range(epochs):
        total_loss = 0  # Initialize total loss for the epoch

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()
            predictions = model(batch_X)
            if(predictions.shape != batch_y.shape):
                print("Mismatched prediction shape",predictions.shape,batch_y.shape)

            loss = criterion(predictions, batch_y)

            loss.backward()
            
            optimizer.step()

            total_loss += loss.item()  # Accumulate the loss for each batch

        average_loss = total_loss / len(train_loader)  # Compute the average loss for the epoch   
        if epoch % 20 == 0: 
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {average_loss:.4f}')

def compute_eval_metrics(y_pred, y_test):
    """
    Calculates MAE, MSE, and Cosine Similarity.

    Args:
        y_pred: The predicted output by the Regression model for the test data.
        y_test: The expected output from the test data.

    Returns:
        (mae, mse, cosine_sim): A tuple containing the computed Mean Absolute Error, Mean Squared Error, and the Cosine Similarity.
    """
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test)

    mae_loss_fn = nn.L1Loss()
    mae = mae_loss_fn(y_pred, y_test)

    mse_loss_fn = nn.MSELoss()
    mse = mse_loss_fn(y_pred, y_test)

    cosine_sim = torch.mean(functional.cosine_similarity(y_pred, y_test, dim=1))
    
    return mae, mse, cosine_sim

def load_model_features(filelist, feature_directory):
    """
    Loads features from .npy files for a given list of filenames.

    Args:
        feature_directory: Directory where the feature files are stored.
        filelist: List of filenames for which to load the features.
    Returns:
        A list of loaded feature arrays.
    """
    features = []
    for filename in filelist:
        feature_path = os.path.join(feature_directory, filename.replace('.wav', '_features.npy'))
        if os.path.exists(feature_path):
            feature_array = np.load(feature_path)
            features.append(feature_array)
        else:
            print(f"Warning: Feature file {feature_path} does not exist. Skipping.")
    return features
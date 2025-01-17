import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

class IOTDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (np.ndarray): The input features of shape (num_samples, features).
            labels (np.ndarray or pd.Series): The labels corresponding to the input features of shape (num_samples,).
        """
        # Convert features and labels to numpy if needed
        if not isinstance(features, np.ndarray):
            features = features.to_numpy()
        if not isinstance(labels, np.ndarray):
            labels = labels.to_numpy()

        # Convert numpy.ndarray to torch.Tensor
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Reshape features if 2D: (num_samples, features) -> (num_samples, seq_length, input_size)
        if features.ndimension() == 2:
            self.features = features.unsqueeze(1)  # Convert to (num_samples, seq_length=1, input_size=features)
        elif features.ndimension() == 3:
            self.features = features  # Assume already (num_samples, seq_length, input_size)
        else:
            raise ValueError("Unexpected feature shape. Expected 2D or 3D tensor.")
        
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SimpleCNN(nn.Module):
    def __init__(self, input_size,num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size of the input to the fully connected layer
        conv_output_size = input_size // 2
        
        self.fc1 = nn.Linear(16 * conv_output_size, 64)  # Adjust for pooling
        self.relu2 = nn.ReLU()

        self.fc2 = nn.Linear(64, num_classes)  # Assuming 10 classes; adjust if needed

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x



def train_model(X_train_selected,X_test_selected,X_val_selected, y_train,y_test,y_val, classification_type):
    print("....MODEL TRAINING STARTED....")
    
    # Example Initialization (Replace with your actual data)
    train_dataset = IOTDataset(X_train_selected, y_train)
    val_dataset = IOTDataset(X_val_selected, y_val)
    test_dataset = IOTDataset(X_test_selected, y_test)

    # Define Batch Size
    BATCH_SIZE = 64

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Define hyperparameters
    input_size = 30  # Length of the feature vector
    num_epochs = 50
    learning_rate = 0.001

    if classification_type=="b":
        num_classes = 2
        model = SimpleCNN(input_size=input_size, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
    
    elif classification_type=="m":
        num_classes = 7
        model = SimpleCNN(input_size=input_size, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        epoch_train_correct = 0
        epoch_train_total = 0

        # Training phase
        for features, labels in train_loader:
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Store training loss
            epoch_train_losses.append(loss.item())
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            epoch_train_total += labels.size(0)
            epoch_train_correct += (predicted == labels).sum().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate epoch metrics for training
        epoch_train_loss = np.mean(epoch_train_losses)
        epoch_train_acc = epoch_train_correct / epoch_train_total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation phase
        model.eval()
        epoch_val_losses = []
        epoch_val_correct = 0
        epoch_val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                epoch_val_losses.append(loss.item())
                
                _, predicted = torch.max(outputs.data, 1)
                epoch_val_total += labels.size(0)
                epoch_val_correct += (predicted == labels).sum().item()

        # Calculate epoch metrics for validation
        epoch_val_loss = np.mean(epoch_val_losses)
        epoch_val_acc = epoch_val_correct / epoch_val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    # Plot training and validation metrics
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Final test evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.2%}")

    

    print("....MODEL TRAINING COMPLETED....")

    return all_labels,all_preds
    
    
    
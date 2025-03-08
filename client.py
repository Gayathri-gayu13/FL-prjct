import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Load ADFA-LD Dataset
# -------------------------------

def load_adfa_data(dataset_path):
    sequences = []
    labels = []

    normal_dir = os.path.join(dataset_path, "Training_Data_Master")
    attack_dir = os.path.join(dataset_path, "Attack_Data_Master")

    # Load normal data (label = 0)
    for file in os.listdir(normal_dir):
        with open(os.path.join(normal_dir, file), "r") as f:
            sequences.append(f.read())
            labels.append(0)  # Normal data = label 0

    # Load attack data (label = 1)
    attack_count = 0  # Debugging: Count attack samples
    for attack_type in os.listdir(attack_dir):  # Loop through attack types
        attack_path = os.path.join(attack_dir, attack_type)
        for file in os.listdir(attack_path):
            with open(os.path.join(attack_path, file), "r") as f:
                sequences.append(f.read())
                labels.append(1)  # Attack data = label 1
                attack_count += 1

    print(f"✅ Loaded {len(sequences)} total samples.")
    print(f"🔍 Attack Samples Count: {attack_count}")  # Debugging output

    return sequences, labels


# -------------------------------
# Convert ADFA-LD to TF-IDF Features
# -------------------------------

def preprocess_data(sequences, labels, batch_size=50000):
    """
    Converts system call sequences to optimized TF-IDF features in batches.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
    
    # Process in smaller batches
    X_sparse = vectorizer.fit_transform(sequences[:batch_size])  # Only process a subset
    X = X_sparse.astype(np.float32).toarray()  # Convert sparse to dense in float32

    y = np.array(labels[:batch_size], dtype=np.int64)  # Only take a subset of labels
    unique, counts = np.unique(y, return_counts=True)
    print(f"🔍 Label Distribution: {dict(zip(unique, counts))}")  # Check how many 0s and 1s

    return X, y, vectorizer
    



# -------------------------------
# Define MLP Model
# -------------------------------

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# -------------------------------
# Load ADFA-LD Data and Prepare Dataset
# -------------------------------

dataset_path = "ADFA-LD"  # Update this with your actual dataset path
sequences, labels = load_adfa_data(dataset_path)
X, y, vectorizer = preprocess_data(sequences, labels)

# Convert to PyTorch tensors
# Ensure X is a dense NumPy array before converting to a tensor
X_dense = np.array(X, dtype=np.float32)  # Explicitly convert to float32
X_tensor = torch.tensor(X_dense)  # Now it's safe for PyTorch

y_tensor = torch.tensor(y, dtype=torch.long)  # Labels as tensors
dataset = TensorDataset(X_tensor, y_tensor)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -------------------------------
# Federated Learning Client
# -------------------------------

class FLClient(fl.client.NumPyClient):
    def __init__(self):
        input_size = X.shape[1]  # TF-IDF feature size
        self.model = MLP(input_size=input_size, hidden_size=32, output_size=2)

        # 🔹 Add missing train_loader initialization
        global train_loader
        self.train_loader = train_loader  # Now train_loader is accessible in `fit()`


    def get_parameters(self, config=None):
        return [param.detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)

    def train_model(self, epochs=1):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total_samples = 0, 0, 0

        self.model.train()
        for X_batch, y_batch in self.train_loader:
            optimizer.zero_grad()
            outputs = self.model(X_batch)

        # 🔹 Debugging: Check if y_batch is valid
            if y_batch is None or len(y_batch) == 0:
                print("❌ ERROR: y_batch is empty!")
                continue  # Skip this batch

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Compute training accuracy
            total_loss += loss.item() * y_batch.size(0)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = correct / total_samples if total_samples > 0 else 0

        print(f"✅ Training Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}")  # Debugging Output
        return self.get_parameters(), len(self.train_loader.dataset), {"loss": avg_loss, "accuracy": avg_accuracy}






    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        correct, total = 0, 0
        all_preds = []  # Store predictions for debugging
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                outputs = self.model(X_batch)
                predicted = torch.argmax(outputs, dim=1)

                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

                all_preds.extend(predicted.cpu().numpy())  # Store predictions
                all_labels.extend(y_batch.cpu().numpy())  # Store actual labels

        accuracy = correct / total if total > 0 else 0  
        print(f"✅ Evaluation Accuracy: {accuracy:.4f}")

    # 🔹 Ensure predictions and labels are printed
        if len(all_preds) > 0 and len(all_labels) > 0:
            print(f"🔍 Sample Predictions: {all_preds[:20]}")  # Print first 20 predictions
            print(f"🔍 Sample Labels: {all_labels[:20]}")  # Print first 20 actual labels")
        else:
            print("❌ ERROR: No predictions or labels found!")

        return float(accuracy), len(self.train_loader.dataset), {"accuracy": accuracy}



# -------------------------------
# Start Federated Client
# -------------------------------

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FLClient())

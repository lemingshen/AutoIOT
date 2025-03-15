# HAR_system.py
import pandas as pd
import numpy as np
from scipy import stats, signal
from scipy.fftpack import fft
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import argparse


# Define the human activity recognition model using PyTorch
class HARModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        conv_channels,
        lstm_hidden_size,
        lstm_layers,
        dropout_prob,
    ):
        super(HARModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        input_channels = 1
        for output_channels in conv_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(output_channels),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
            )
            input_channels = output_channels
        self.flattened_size = (
            input_size // (2 ** len(conv_channels)) * conv_channels[-1]
        )
        self.lstm = nn.LSTM(
            input_size=self.flattened_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.reshape(x.size(0), -1, self.flattened_size)
        x, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = self.dropout(x)
        out = self.fc(x)
        return out


# Define the function to load data
def load_data(file_path):
    columns = ["user", "activity", "timestamp", "x-axis", "y-axis", "z-axis"]
    data = []
    with open(file_path) as f:
        for line in f:
            try:
                row = line.strip().split(",")
                if len(row) == 6:
                    data.append(row)
            except:
                continue
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Convert data types
    df["x-axis"] = pd.to_numeric(df["x-axis"], errors="coerce")
    df["y-axis"] = pd.to_numeric(df["y-axis"], errors="coerce")
    df["z-axis"] = pd.to_numeric(df["z-axis"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    # Encode activity labels into numeric format
    label_encoder = LabelEncoder()
    df["activity"] = label_encoder.fit_transform(df["activity"])

    return df


# Define the function to clean data
def clean_data(df):
    df_cleaned = df.copy()
    df_cleaned["x-axis"] = df_cleaned["x-axis"].rolling(window=3).mean()
    df_cleaned["y-axis"] = df_cleaned["y-axis"].rolling(window=3).mean()
    df_cleaned["z-axis"] = df_cleaned["z-axis"].rolling(window=3).mean()
    df_cleaned.dropna(inplace=True)
    return df_cleaned


# Define the function to normalize data
def normalize_data(df):
    df_normalized = df.copy()
    df_normalized[["x-axis", "y-axis", "z-axis"]] = preprocessing.scale(
        df[["x-axis", "y-axis", "z-axis"]]
    )
    return df_normalized


# Define the function to segment data
def segment_data(df, window_size=256, overlap=0.5):
    step = int(window_size * (1 - overlap))
    segments = []
    labels = []
    for i in range(0, len(df) - window_size, step):
        xs = df["x-axis"].values[i : i + window_size]
        ys = df["y-axis"].values[i : i + window_size]
        zs = df["z-axis"].values[i : i + window_size]
        label = stats.mode(df["activity"][i : i + window_size])[0]
        segments.append([xs, ys, zs])
        labels.append(label)
    return segments, labels


# Define the function to extract features
def extract_features(segments):
    features = []
    for segment in segments:
        current_features = np.array(
            [
                np.mean(segment[0]),
                np.mean(segment[1]),
                np.mean(segment[2]),
                np.var(segment[0]),
                np.var(segment[1]),
                np.var(segment[2]),
                np.max(segment[0]),
                np.max(segment[1]),
                np.max(segment[2]),
                np.min(segment[0]),
                np.min(segment[1]),
                np.min(segment[2]),
            ]
        )
        fft_features = np.concatenate(
            [
                np.abs(fft(segment[0]))[0 : int(len(segment[0]) / 2)],
                np.abs(fft(segment[1]))[0 : int(len(segment[1]) / 2)],
                np.abs(fft(segment[2]))[0 : int(len(segment[2]) / 2)],
            ]
        )
        features.append(np.concatenate([current_features, fft_features]))

    return np.array(features)


# Define the function to encode labels
def label_encoding(labels):
    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    return encoded_labels


# Define the function to split data
def split_data(features, labels, test_size=0.3, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return X_train, X_test, y_train, y_test


# Define the function to train the model
def train_model(
    model,
    train_features,
    train_labels,
    val_features,
    val_labels,
    batch_size,
    learning_rate,
    epochs,
    device,
):
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct / total
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}"
        )
    model.load_state_dict(best_model_wts)
    return model, best_accuracy


# Define the function to evaluate the model
def evaluate_model(model, test_features, test_labels, batch_size, device):
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


# Define the function to output the result
def output_result(accuracy):
    formatted_accuracy = "{:.2f}".format(accuracy)
    print(f"Average recognition accuracy: {formatted_accuracy}")


# Main function
def main(input_file):
    # Load the dataset
    df = load_data(input_file)
    df_cleaned = clean_data(df)
    df_normalized = normalize_data(df_cleaned)
    segments, labels = segment_data(df_normalized)
    features = extract_features(segments)
    encoded_labels = label_encoding(labels)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(
        features, encoded_labels, test_size=0.3, random_state=42
    )

    # Define the model parameters
    input_size = features.shape[
        1
    ]  # This should be the number of features for each time window
    output_size = len(
        np.unique(encoded_labels)
    )  # This should be the number of activity classes
    conv_channels = [64, 128]  # Example channel sizes for convolutional layers
    lstm_hidden_size = 64  # Example size for LSTM hidden state
    lstm_layers = 2  # Number of LSTM layers
    dropout_prob = 0.5  # Dropout probability
    batch_size = 64  # Define a suitable batch size
    learning_rate = 0.001  # Define a suitable learning rate
    epochs = 10  # Define a suitable number of epochs

    # Instantiate the model and move it to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HARModel(
        input_size,
        output_size,
        conv_channels,
        lstm_hidden_size,
        lstm_layers,
        dropout_prob,
    )
    model.to(device)

    # Train the model
    model, best_accuracy = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size,
        learning_rate,
        epochs,
        device,
    )

    # Evaluate the model
    test_accuracy = evaluate_model(model, X_test, y_test, batch_size, device)

    # Output the result
    output_result(test_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Activity Recognition System")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to the WISDM dataset file"
    )
    args = parser.parse_args()

    main(args.input)

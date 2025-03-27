import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Load multimodal data from npy files and verify their integrity.
def load_multimodal_data(dataset_path):
    train_audio_data = np.load(f"{dataset_path}/train_audio.npy")
    train_depth_data = np.load(f"{dataset_path}/train_depth.npy")
    train_radar_data = np.load(f"{dataset_path}/train_radar.npy")
    train_label = np.load(f"{dataset_path}/train_label.npy")
    test_audio_data = np.load(f"{dataset_path}/test_audio.npy")
    test_depth_data = np.load(f"{dataset_path}/test_depth.npy")
    test_radar_data = np.load(f"{dataset_path}/test_radar.npy")
    test_label = np.load(f"{dataset_path}/test_label.npy")

    train_audio_data = torch.from_numpy(train_audio_data).float()
    train_depth_data = torch.from_numpy(train_depth_data).float()
    train_radar_data = torch.from_numpy(train_radar_data).float()
    train_label = torch.from_numpy(train_label).long()
    test_audio_data = torch.from_numpy(test_audio_data).float()
    test_depth_data = torch.from_numpy(test_depth_data).float()
    test_radar_data = torch.from_numpy(test_radar_data).float()
    test_label = torch.from_numpy(test_label).long()

    train_loader = DataLoader(
        dataset=TensorDataset(
            train_audio_data, train_depth_data, train_radar_data, train_label
        ),
        batch_size=32,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=TensorDataset(
            test_audio_data, test_depth_data, test_radar_data, test_label
        ),
        batch_size=32,
        shuffle=False,
    )

    return train_loader, test_loader


class Audio_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.projection = nn.Linear(32 * 20 * 87, 256)

    def forward(self, x):
        # the shape of x should be (batch_size, 1, 20, 87)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 20, 87)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        feature = self.projection(x.view(batch_size, 32 * 20 * 87))

        return feature


class Depth_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.projection = nn.Linear(64 * 112 * 112, 256)

    def forward(self, x):
        # the shape of x should be (batch_size, 16, 112, 112)
        batch_size = x.size(0)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        feature = self.projection(x.view(batch_size, 64 * 112 * 112))

        return feature


class Radar_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.projection = nn.Linear(64 * 2 * 16 * 32 * 16, 256)

    def forward(self, x):
        # the shape of x should be (batch_size, 20, 2 * 16, 32 * 16)
        batch_size = x.size(0)
        x = x.view(batch_size, 20, 2 * 16, 32 * 16)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        feature = self.projection(x.view(batch_size, 64 * 2 * 16 * 32 * 16))

        return feature


class MultimodalActivityRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.audio_encoder = Audio_Encoder()
        self.depth_encoder = Depth_Encoder()
        self.radar_encoder = Radar_Encoder()

        self.fc1 = nn.Linear(256 * 3, 256)
        self.fc2 = nn.Linear(256, 11)

    def forward(self, audio, depth, radar):
        audio_feature = self.audio_encoder(audio)
        depth_feature = self.depth_encoder(depth)
        radar_feature = self.radar_encoder(radar)

        x = torch.cat((audio_feature, depth_feature, radar_feature), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Train the multimodal model using the training data and validate using validation data.  
def train_multimodal_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10,
    learning_rate=0.001,
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_test_loss = float("inf")
    patience, patience_counter = 10, 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for audio_batch, depth_batch, radar_batch, labels in train_loader:
            audio_batch, depth_batch, radar_batch, labels = (
                audio_batch.to(device),
                depth_batch.to(device),
                radar_batch.to(device),
                labels.to(device),
            )

            outputs = model(audio_batch, depth_batch, radar_batch)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for audio_batch, depth_batch, radar_batch, labels in test_loader:
                audio_batch, depth_batch, radar_batch, labels = (
                    audio_batch.to(device),
                    depth_batch.to(device),
                    radar_batch.to(device),
                    labels.to(device),
                )

                outputs = model(audio_batch, depth_batch, radar_batch)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        scheduler.step(test_loss)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.  4f}, Val Loss: {test_loss/len(test_loader):.4f}"
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return model


# Evaluate the trained model on the test dataset and calculate performance metrics.
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct, total = 0, 0
    y_test_list, y_pred_list = [], []
    with torch.no_grad():
        for audio_test, depth_test, radar_test, y_test in test_loader:
            audio_test, depth_test, radar_test, y_test = (
                audio_test.to(device),
                depth_test.to(device),
                radar_test.to(device),
                y_test.to(device),
            )
            outputs = model(audio_test, depth_test, radar_test)
            _, predicted = torch.max(outputs.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()
            y_test_list.extend(list(y_test.cpu().numpy()))
            y_pred_list.extend((predicted.cpu().numpy()))

    accuracy = correct / total
    conf_matrix = confusion_matrix(y_test_list, y_pred_list)

    return accuracy, conf_matrix


# Output the average recognition accuracy and visualize the model's performance.
def output_results(accuracy, conf_matrix):
    print(f"Average recognition accuracy on test data: {accuracy * 100:.2f}%")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def main(dataset_path):
    # Load the multimodal dataset
    train_loader, test_loader = load_multimodal_data(dataset_path)

    # Create a model instance
    model = MultimodalActivityRecognitionModel()

    # Train the model
    trained_model = train_multimodal_model(model, train_loader, test_loader)

    # Evaluate the model
    accuracy, conf_matrix = evaluate_model(trained_model, test_loader)

    # Output the results
    output_results(accuracy, conf_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal Human Activity Recognition"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the   dataset")
    args = parser.parse_args()
    main(args.input)
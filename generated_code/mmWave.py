import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import argparse


# Custom Dataset class for XRF55 dataset to work with PyTorch DataLoader
class XRF55Dataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, label_tensor: torch.Tensor):
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index]


# Custom CNN architecture for human motion recognition
class CustomCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                17, 64, kernel_size=3, padding=1
            ),  # Updated to accept 17 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 16 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss criterion.
    """

    def __init__(self, classes, smoothing=0.5):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        """
        Compute the label smoothing loss.

        Args:
            pred (Tensor): Predicted probabilities of shape (batch_size, num_classes).
            target (Tensor): Target labels of shape (batch_size,).

        Returns:
            Tensor: Computed label smoothing loss.
        """
        assert 0 <= self.smoothing < 1
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smoothed_labels = one_hot * (1 - self.smoothing) + torch.full_like(
            pred, self.smoothing / self.classes
        )
        log_prob = torch.nn.functional.log_softmax(pred, dim=-1)
        return torch.mean(torch.sum(-smoothed_labels * log_prob, dim=-1))


def split_dataset_and_create_dataloaders(
    data_tensor, label_tensor, batch_size=64, train_split=0.8
):
    dataset = XRF55Dataset(data_tensor, label_tensor)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# Function to load and preprocess the XRF55 dataset
def load_and_preprocess_xrf55_dataset(dataset_folder_path):
    activity_id_to_class_label = {
        "45": 0,
        "46": 1,
        "47": 2,
        "48": 3,
        "49": 4,
        "50": 5,
        "51": 6,
        "52": 7,
        "53": 8,
        "54": 9,
        "55": 10,
    }
    data_list = []
    label_list = []

    dirs = os.listdir(dataset_folder_path)
    for dir in dirs:
        files = "{}/{}".format(dataset_folder_path, dir)
        for file in os.listdir(files):
            if file.endswith(".npy"):
                file_name_parts = file.split("_")
                user_id, activity_id, trial_number = (
                    file_name_parts[0],
                    file_name_parts[1],
                    file_name_parts[2],
                )
                if activity_id in activity_id_to_class_label:
                    file_path = os.path.join(dataset_folder_path, dir, file)
                    heatmap = np.load(file_path)
                    # Reshape to remove the first dimension of size 1
                    heatmap = heatmap.reshape((17, 256, 128))
                    heatmap_normalized = heatmap / np.max(heatmap)
                    heatmap_tensor = torch.tensor(
                        heatmap_normalized, dtype=torch.float32
                    )
                    data_list.append(heatmap_tensor)
                    label_list.append(activity_id_to_class_label[activity_id])
    # Check if the data list is not empty before stacking
    if data_list:
        data_tensor = torch.stack(data_list)
        label_tensor = torch.tensor(label_list, dtype=torch.long)
        return data_tensor, label_tensor
    else:
        raise RuntimeError("No data found in the specified dataset folder path.")


def mixup_data(x, y, alpha=1.0):
    # Randomly generate parameters lam of a beta distribution to generate random linear combinations to implement mixup data augmentation
    lam = np.random.beta(alpha, alpha)
    # Generate a random sequence for shuffling the input data.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    # Get a new mixed data
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # Get the two types of labels corresponding to the mixed image
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# Function to train the model with regularization and scheduling
def train_with_regularization_and_scheduling(
    model, train_loader, val_loader, device, epochs=25, lr=0.001
):
    """
    Trains the model using regularization techniques and adaptive learning rate scheduling.

    Parameters:
    model (nn.Module): The neural network model to train.
    train_loader (DataLoader): DataLoader for the training set.
    val_loader (DataLoader): DataLoader for the validation set.
    device (torch.device): The device to train the model on (CPU or GPU).
    epochs (int): Number of epochs to train the model.
    lr (float): Initial learning rate for the optimizer.

    Returns:
    None
    """
    # Define the loss function with label smoothing
    criterion = LabelSmoothingLoss(classes=11, smoothing=0.1)
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Define the learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=10, verbose=True
    )

    # Initialize variables for early stopping
    best_val_accuracy = 0
    epochs_no_improve = 0
    early_stop_threshold = 10

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Perform mixup augmentation
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
            inputs, targets_a, targets_b = map(
                torch.autograd.Variable, (inputs, targets_a, targets_b)
            )

            # Forward pass
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                outputs, targets_b
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        val_accuracy = evaluate_model_performance(model, val_loader)
        print(f"Epoch {epoch+1}: Validation Accuracy: {val_accuracy}")

        # Learning rate scheduling
        scheduler.step(val_accuracy)

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_threshold:
                print("Early stopping triggered!")
                break


# Function to evaluate the model's performance
def evaluate_model_performance(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted_classes = torch.max(outputs, 1)
            correct_predictions += (predicted_classes == target).sum().item()
            total_predictions += target.size(0)
    average_accuracy = correct_predictions / total_predictions

    return average_accuracy


def main(dataset_folder_path):
    # Load and preprocess the dataset
    data_tensor, label_tensor = load_and_preprocess_xrf55_dataset(dataset_folder_path)

    # Split the dataset into training and testing sets and create DataLoader instances
    train_loader, test_loader = split_dataset_and_create_dataloaders(
        data_tensor, label_tensor
    )

    # Initialize the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = CustomCNN(num_classes=11).to(device)

    # Train the model
    train_with_regularization_and_scheduling(model, train_loader, test_loader, device)

    # Evaluate the model
    val_accuracy = evaluate_model_performance(model, test_loader)
    print(f"Final Validation Accuracy: {val_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Motion Recognition System")
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the dataset folder"
    )
    args = parser.parse_args()
    main(args.input)

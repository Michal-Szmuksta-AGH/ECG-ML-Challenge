import os
import sys
import wandb
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from loguru import logger
from datetime import datetime
from torch.utils.data import DataLoader

from src.config import MODELS_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR
from src.dataset.dataloaders import ECGDataset
from src.model.models import get_model


class DiceLoss1D(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss1D, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        intersection = torch.sum(targets * preds)
        union = torch.sum(targets) + torch.sum(preds)
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        return dice_loss


class CombinedDiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.2):
        super(CombinedDiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = 1 - dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss1D()

    def forward(self, preds, targets):
        bce_loss = self.bce_loss(preds, targets)
        dice_loss = self.dice_loss(preds, targets)
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return combined_loss


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Train the model for one epoch.

    :param model: The model to train.
    :param train_loader: DataLoader for the training data.
    :param criterion: Loss function.
    :param optimizer: Optimizer for updating model parameters.
    :param device: Device to perform training on (CPU or GPU).
    """
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(device).cuda(non_blocking=True), y.to(device).float().cuda(non_blocking=True)

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Training Loss: {avg_loss:.4f}")
    wandb.log({"Training Loss": avg_loss})


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    prefix: str = "Test",
) -> None:
    """
    Evaluate the model on the given data.

    :param model: The model to evaluate.
    :param data_loader: DataLoader for the data.
    :param criterion: Loss function.
    :param device: Device to perform evaluation on (CPU or GPU).
    :param prefix: Prefix for logging (e.g., "Test" or "Train").
    """
    model.eval()
    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    dice_metric = torchmetrics.Dice().to(device)
    total_loss = 0
    calculate_loss = prefix != "Train"

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {prefix}"):
            x, y = batch
            x, y = x.to(device), y.to(device).float()

            outputs = model(x)
            if calculate_loss:
                loss = criterion(outputs, y)
                total_loss += loss.item()
            preds = (outputs > 0.5).float()

            accuracy_metric.update(preds, y.int())
            precision_metric.update(preds, y.int())
            recall_metric.update(preds, y.int())
            f1_metric.update(preds, y.int())
            dice_metric.update(preds, y.int())

    if calculate_loss:
        avg_loss = total_loss / len(data_loader)
        logger.info(f"{prefix} Loss: {avg_loss:.4f}")
        wandb.log({f"{prefix} Loss": avg_loss})

    accuracy = accuracy_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    dice = dice_metric.compute().item()

    logger.info(f"{prefix} Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"{prefix} Precision: {precision:.4f}")
    logger.info(f"{prefix} Recall: {recall:.4f}")
    logger.info(f"{prefix} F1 Score: {f1:.4f}")
    logger.info(f"{prefix} Dice Score: {dice:.4f}")

    wandb.log(
        {
            f"{prefix} Accuracy": accuracy,
            f"{prefix} Precision": precision,
            f"{prefix} Recall": recall,
            f"{prefix} F1 Score": f1,
            f"{prefix} Dice Score": dice,
        }
    )


def save_model_and_report(
    model: nn.Module, model_type: str, epochs: int, batch_size: int, learning_rate: float
) -> None:
    """
    Save the model and wandb report with a specific naming convention.

    :param model: The trained model.
    :param model_type: Type of the model.
    :param epochs: Number of epochs the model was trained for.
    :param batch_size: Batch size used during training.
    :param learning_rate: Learning rate used during training.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f"{model_type}_{timestamp}_{epochs}_{batch_size}_{learning_rate}.pth"
    model_save_path = os.path.join(MODELS_DIR, model_filename)

    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    artifact = wandb.Artifact(name=model_filename, type="model")
    artifact.add_file(model_save_path)
    wandb.log_artifact(artifact)
    logger.info(f"Model artifact logged to wandb")


def train_model(
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    model_type: str = "LSTMModel",
    train_data_dir: str = TRAIN_DATA_DIR,
    val_data_dir: str = VAL_DATA_DIR,
    verbosity: str = "INFO",
    resume_model: str = None,
) -> None:
    """
    Train the model with the specified options.

    :param epochs: Number of epochs to train.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for the optimizer.
    :param model_type: Type of model to use.
    :param train_data_dir: Directory for training data.
    :param test_data_dir: Directory for test data.
    :param val_data_dir: Directory for validation data.
    :param verbosity: Logging verbosity level.
    :param resume_model: Path to a saved model to resume training from. If None, train from scratch.
    """
    wandb.init(
        project="ECG-ML-Challenge",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_type": model_type,
        },
    )
    logger.remove()
    logger.add(sys.stderr, level=verbosity)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_dataset = ECGDataset(directory=train_data_dir)
    val_dataset = ECGDataset(directory=val_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = get_model(model_type)
    if resume_model:
        model.load_state_dict(torch.load(resume_model))
        logger.info(f"Resumed training from model: {resume_model}")
    model = torch.compile(model)
    model = model.to(device)

    # pos_weight = torch.tensor([3.7]).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = CombinedDiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(wandb.config.epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        wandb.log({"Epoch": epoch + 1})
        train(model, train_loader, criterion, optimizer, device)
        evaluate(model, train_loader, criterion, device, prefix="Train")
        evaluate(model, val_loader, criterion, device, prefix="Validation")

    save_model_and_report(model, model_type, epochs, batch_size, learning_rate)
    wandb.finish()

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def get_model(model_type: str, *args, **kwargs):
    """
    Get the model class based on the model type.

    :param model_type: Type of model to get.
    """
    model_classes = {
        name: cls
        for name, cls in globals().items()
        if inspect.isclass(cls) and issubclass(cls, nn.Module)
    }
    if model_type in model_classes:
        return model_classes[model_type](*args, **kwargs)
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")


class LSTMModel(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the LSTMModel.
        """
        super(LSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(64 * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LSTMModel.

        :param x: Input tensor of shape [batch_size, seq_length].
        :return: Output tensor of shape [batch_size, seq_length].
        """
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x.squeeze(-1)


class MultiScaleConvLSTMModel(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the MultiScaleConvLSTMModel.
        """
        super(MultiScaleConvLSTMModel, self).__init__()
        # First scale
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm1d(16)
        self.conv1_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm1d(32)

        # Second scale
        self.conv2_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn2_1 = nn.BatchNorm1d(16)
        self.conv2_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2_2 = nn.BatchNorm1d(32)

        # Third scale
        self.conv3_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        self.bn3_1 = nn.BatchNorm1d(16)
        self.conv3_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.bn3_2 = nn.BatchNorm1d(32)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=96, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(64 * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MultiScaleConvLSTMModel.

        :param x: Input tensor of shape [batch_size, seq_length].
        :return: Output tensor of shape [batch_size, seq_length].
        """
        x = x.unsqueeze(1)  # Add channel dimension, shape: [batch_size, 1, seq_length]
        seq_length = x.size(-1)  # Pobierz dynamiczną długość wejścia

        # First scale
        x1 = torch.relu(self.bn1_1(self.conv1_1(x)))  # [batch_size, 16, seq_length]
        x1 = torch.nn.functional.adaptive_max_pool1d(x1, output_size=seq_length)
        x1 = torch.relu(self.bn1_2(self.conv1_2(x1)))  # [batch_size, 32, seq_length]
        x1 = torch.nn.functional.adaptive_max_pool1d(x1, output_size=seq_length)

        # Second scale
        x2 = torch.relu(self.bn2_1(self.conv2_1(x)))  # [batch_size, 16, seq_length]
        x2 = torch.nn.functional.adaptive_max_pool1d(x2, output_size=seq_length)
        x2 = torch.relu(self.bn2_2(self.conv2_2(x2)))  # [batch_size, 32, seq_length]
        x2 = torch.nn.functional.adaptive_max_pool1d(x2, output_size=seq_length)

        # Third scale
        x3 = torch.relu(self.bn3_1(self.conv3_1(x)))  # [batch_size, 16, seq_length]
        x3 = torch.nn.functional.adaptive_max_pool1d(x3, output_size=seq_length)
        x3 = torch.relu(self.bn3_2(self.conv3_2(x3)))  # [batch_size, 32, seq_length]
        x3 = torch.nn.functional.adaptive_max_pool1d(x3, output_size=seq_length)

        # Concatenate along the channel dimension
        x = torch.cat(
            (x1, x2, x3), dim=1
        )  # Concatenate outputs, shape: [batch_size, 96, seq_length]

        # Dropout
        x = self.dropout(x)

        # LSTM
        x = x.transpose(1, 2)  # Change shape to [batch_size, seq_length, features]
        x, _ = self.lstm(x)
        x = self.fc(x)  # Fully connected layer
        x = torch.sigmoid(x)

        return x.squeeze(-1)  # Output shape: [batch_size, seq_length]


class GPTMultiScaleConvGRUModel(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the GPTMultiScaleConvGRUModel.
        """
        super(GPTMultiScaleConvGRUModel, self).__init__()

        # First scale
        self.conv1_1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm1d(16)
        self.conv1_2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.conv1_3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn1_3 = nn.BatchNorm1d(64)
        self.shortcut1 = nn.Conv1d(1, 64, kernel_size=1, padding=0)

        # Second scale
        self.conv2_1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn2_1 = nn.BatchNorm1d(16)
        self.conv2_2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.conv2_3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2_3 = nn.BatchNorm1d(64)
        self.shortcut2 = nn.Conv1d(1, 64, kernel_size=1, padding=0)

        # Third scale
        self.conv3_1 = nn.Conv1d(1, 16, kernel_size=15, padding=7)
        self.bn3_1 = nn.BatchNorm1d(16)
        self.conv3_2 = nn.Conv1d(16, 32, kernel_size=15, padding=7)
        self.bn3_2 = nn.BatchNorm1d(32)
        self.conv3_3 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        self.bn3_3 = nn.BatchNorm1d(64)
        self.shortcut3 = nn.Conv1d(1, 64, kernel_size=1, padding=0)

        # Layer normalization
        self.norm1 = nn.LayerNorm(192)  # Adjusted for increased channels

        # GRU layer
        self.gru = nn.GRU(
            input_size=192, hidden_size=128, num_layers=3, batch_first=True, bidirectional=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GPTMultiScaleConvGRUModel.

        :param x: Input tensor of shape [batch_size, seq_length].
        :return: Output tensor of shape [batch_size, seq_length].
        """
        x = x.unsqueeze(1)

        # First scale with skip connection
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
        x1 = F.relu(self.bn1_3(self.conv1_3(x1) + self.shortcut1(x)))  # Skip connection
        x1 = F.adaptive_max_pool1d(x1, output_size=x.size(-1))

        # Second scale with skip connection
        x2 = F.relu(self.bn2_1(self.conv2_1(x)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x2 = F.relu(self.bn2_3(self.conv2_3(x2) + self.shortcut2(x)))  # Skip connection
        x2 = F.adaptive_max_pool1d(x2, output_size=x.size(-1))

        # Third scale with skip connection
        x3 = F.relu(self.bn3_1(self.conv3_1(x)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = F.relu(self.bn3_3(self.conv3_3(x3) + self.shortcut3(x)))  # Skip connection
        x3 = F.adaptive_max_pool1d(x3, output_size=x.size(-1))

        # Concatenate features from all scales
        x = torch.cat((x1, x2, x3), dim=1)  # Shape: [batch_size, 192, seq_length]

        # Layer normalization
        x = x.transpose(1, 2)  # Shape: [batch_size, seq_length, 192]
        x = self.norm1(x)

        # GRU
        x, _ = self.gru(x)  # Output shape: [batch_size, seq_length, 256]

        # Fully connected layers
        x = F.relu(self.fc1(x))  # Shape: [batch_size, seq_length, 64]
        x = self.fc2(x)  # Shape: [batch_size, seq_length, 1]
        # x = torch.sigmoid(x)

        return x.squeeze(-1)  # Output shape: [batch_size, seq_length]


class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int = 1000,
        hidden_size: int = 16,
        num_layers: int = 1,
        output_size: int = 1000,
    ) -> None:
        """
        Initialize the GRUModel.

        :param input_size: Size of the input features.
        :param hidden_size: Number of features in the hidden state.
        :param num_layers: Number of recurrent layers.
        :param output_size: Size of the output features.
        """
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GRUModel.

        :param x: Input tensor of shape [batch_size, seq_length, input_size].
        :return: Output tensor of shape [batch_size, seq_length, output_size].
        """
        out, _ = self.gru(x)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

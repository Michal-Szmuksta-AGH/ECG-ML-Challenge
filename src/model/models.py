import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from src.config import TRAIN_DATA_DIR
import os
import random
import numpy as np


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
        model_class = model_classes[model_type]
        if "input_length" in inspect.signature(model_class).parameters:
            sample_file = random.choice(os.listdir(TRAIN_DATA_DIR))
            sample_path = os.path.join(TRAIN_DATA_DIR, sample_file)
            sample_data = np.load(sample_path)
            input_length = sample_data["x"].shape[-1]
            kwargs["input_length"] = input_length
        return model_class(*args, **kwargs)
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")


class Hook:
    def __init__(self, module, backward=False):
        self.module = module
        self.input = None
        self.output = None
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def add_hooks(model):
    hooks = []
    for layer in model.children():
        hooks.append(Hook(layer))
        if hasattr(layer, "children"):
            hooks.extend(add_hooks(layer))
    return hooks


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


class GPTMultiScaleConvLSTMModelv2(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the GPTMultiScaleConvGRUModel.
        """
        super(GPTMultiScaleConvLSTMModelv2, self).__init__()

        # First scale
        self.conv1_1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm1d(16)
        self.conv1_2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.conv1_3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn1_3 = nn.BatchNorm1d(64)
        self.shortcut1 = nn.Conv1d(1, 64, kernel_size=1, padding=0)

        # Second scale
        self.conv2_1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn2_1 = nn.BatchNorm1d(16)
        self.conv2_2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.conv2_3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2_3 = nn.BatchNorm1d(64)
        self.shortcut2 = nn.Conv1d(1, 64, kernel_size=1, padding=0)

        # Third scale
        self.conv3_1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn3_1 = nn.BatchNorm1d(16)
        self.conv3_2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.bn3_2 = nn.BatchNorm1d(32)
        self.conv3_3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn3_3 = nn.BatchNorm1d(64)
        self.shortcut3 = nn.Conv1d(1, 64, kernel_size=1, padding=0)

        # Layer normalization
        self.norm1 = nn.LayerNorm(192)  # Adjusted for increased channels

        # GRU layer
        self.gru = nn.LSTM(
            input_size=192, hidden_size=512, num_layers=3, batch_first=True, bidirectional=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2, 64)
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


class Deep1DCNN(nn.Module):
    def __init__(self, input_length):
        super(Deep1DCNN, self).__init__()

        # Nowa warstwa konwolucyjna z jeszcze większym kernel size
        self.conv0 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=21, stride=1, padding=10
        )
        self.bn0 = nn.BatchNorm1d(32)
        self.pool0 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Nowa warstwa konwolucyjna z większym kernel size
        self.conv1 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=15, stride=1, padding=7
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Blok 1: Warstwy konwolucyjne + MaxPooling po każdej warstwie
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2
        )
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm1d(512)

        # Blok 2: Ostatnie 3 warstwy konwolucyjne bez pooling
        self.conv5 = nn.Conv1d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.bn5 = nn.BatchNorm1d(512)

        self.conv6 = nn.Conv1d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.bn6 = nn.BatchNorm1d(512)

        self.conv7 = nn.Conv1d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.bn7 = nn.BatchNorm1d(512)

        # Blok 3: Warstwy liniowe
        reduced_length = input_length // (2**4)
        self.fc1 = nn.Linear(512 * reduced_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, input_length)

    def forward(self, x):
        x = x.unsqueeze(1)

        # Nowa warstwa konwolucyjna z jeszcze większym kernel size
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.pool0(x)

        # Nowa warstwa konwolucyjna z większym kernel size
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Blok 1: Warstwy konwolucyjne + MaxPooling po każdej warstwie
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))

        # Blok 2: Ostatnie 3 warstwy konwolucyjne bez pooling
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        # Flatten (przygotowanie danych do warstw w pełni połączonych)
        x = x.view(x.size(0), -1)  # Rzutowanie na [B, feature_size]

        # Blok 3: Warstwy liniowe
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Dodanie wymiaru na końcu, by uzyskać [B, input_length, 1]
        x = x.squeeze(-1)

        return x


class UNet1D(nn.Module):
    def __init__(self, input_channels=1):
        super(UNet1D, self).__init__()
        self.encoder1 = self.conv_block(input_channels, 8)
        self.encoder2 = self.conv_block(8, 16)
        self.encoder3 = self.conv_block(16, 32)
        self.encoder4 = self.conv_block(32, 64)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = self.conv_block(64, 96)

        self.upconv4 = nn.ConvTranspose1d(96, 64, kernel_size=8, stride=2, padding=3)
        self.decoder4 = self.conv_block(64 + 64, 32)
        self.upconv3 = nn.ConvTranspose1d(32, 32, kernel_size=8, stride=2, padding=3)
        self.decoder3 = self.conv_block(32 + 32, 16)
        self.upconv2 = nn.ConvTranspose1d(16, 16, kernel_size=8, stride=2, padding=3)
        self.decoder2 = self.conv_block(16 + 16, 8)
        self.upconv1 = nn.ConvTranspose1d(8, 8, kernel_size=8, stride=2, padding=3)
        self.decoder1 = self.conv_block(8 + 8, 8)

        # Output layer that produces probabilities for each position
        self.final_conv = nn.Conv1d(8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Activation for probability output

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # Encoding path
        x = x.unsqueeze(1)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoding path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Final layer: Single class output (logits)
        logits = self.final_conv(dec1)

        # Apply sigmoid for probabilities
        # output = self.sigmoid(logits)
        output = logits.squeeze(1)

        return output


class GPTMultiScaleConvLSTMModelv2(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the GPTMultiScaleConvGRUModel.
        """
        super(GPTMultiScaleConvLSTMModelv2, self).__init__()

        # First scale
        self.conv1_1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm1d(16)
        self.conv1_2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.conv1_3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn1_3 = nn.BatchNorm1d(64)
        self.shortcut1 = nn.Conv1d(1, 64, kernel_size=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5)

        # Second scale
        self.conv2_1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn2_1 = nn.BatchNorm1d(16)
        self.conv2_2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.conv2_3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2_3 = nn.BatchNorm1d(64)
        self.shortcut2 = nn.Conv1d(1, 64, kernel_size=1, padding=0)
        self.dropout2 = nn.Dropout(p=0.5)

        # Third scale
        self.conv3_1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn3_1 = nn.BatchNorm1d(16)
        self.conv3_2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.bn3_2 = nn.BatchNorm1d(32)
        self.conv3_3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn3_3 = nn.BatchNorm1d(64)
        self.shortcut3 = nn.Conv1d(1, 64, kernel_size=1, padding=0)
        self.dropout3 = nn.Dropout(p=0.5)

        # Layer normalization
        self.norm1 = nn.LayerNorm(192)  # Adjusted for increased channels

        # GRU layer
        self.gru = nn.LSTM(
            input_size=192, hidden_size=512, num_layers=3, batch_first=True, bidirectional=True
        )
        self.dropout_gru = nn.Dropout(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2, 64)
        self.dropout_fc = nn.Dropout(p=0.5)
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
        x1 = self.dropout1(x1)
        x1 = F.adaptive_max_pool1d(x1, output_size=x.size(-1))

        # Second scale with skip connection
        x2 = F.relu(self.bn2_1(self.conv2_1(x)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x2 = F.relu(self.bn2_3(self.conv2_3(x2) + self.shortcut2(x)))  # Skip connection
        x2 = self.dropout2(x2)
        x2 = F.adaptive_max_pool1d(x2, output_size=x.size(-1))

        # Third scale with skip connection
        x3 = F.relu(self.bn3_1(self.conv3_1(x)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = F.relu(self.bn3_3(self.conv3_3(x3) + self.shortcut3(x)))  # Skip connection
        x3 = self.dropout3(x3)
        x3 = F.adaptive_max_pool1d(x3, output_size=x.size(-1))

        # Concatenate features from all scales
        x = torch.cat((x1, x2, x3), dim=1)  # Shape: [batch_size, 192, seq_length]

        # Layer normalization
        x = x.transpose(1, 2)  # Shape: [batch_size, seq_length, 192]
        x = self.norm1(x)

        # GRU
        x, _ = self.gru(x)  # Output shape: [batch_size, seq_length, 256]
        x = self.dropout_gru(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # Shape: [batch_size, seq_length, 64]
        x = self.dropout_fc(x)
        x = self.fc2(x)  # Shape: [batch_size, seq_length, 1]

        return x.squeeze(-1)  # Output shape: [batch_size, seq_length]


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn=nn.ReLU()):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_fn(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.skipconv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.convblock1 = ConvolutionalBlock(in_channels, out_channels)
        self.convblock2 = ConvolutionalBlock(out_channels, out_channels)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        skip = self.skipconv(x)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = x + skip
        x = self.relu1(x)

        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpsampleBlock, self).__init__()
        self.convtrans = nn.ConvTranspose1d(
            in_size,
            out_size,
            kernel_size=3,
            padding=1,
            output_padding=1,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convtrans(x)

        return x


class BestUNet(nn.Module):
    def __init__(self, depth=64):
        super(BestUNet, self).__init__()
        self.residual1 = ResidualBlock(1, depth)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.residual2 = ResidualBlock(depth, depth * 2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.residual3 = ResidualBlock(depth * 2, depth * 4)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.residual4 = ResidualBlock(depth * 4, depth * 8)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.residual5 = ResidualBlock(depth * 8, depth * 16)
        self.upsample1 = UpsampleBlock(depth * 16, depth * 8)
        self.residual6 = ResidualBlock(depth * 16, depth * 8)
        self.upsample2 = UpsampleBlock(depth * 8, depth * 4)
        self.residual7 = ResidualBlock(depth * 8, depth * 4)
        self.upsample3 = UpsampleBlock(depth * 4, depth * 2)
        self.residual8 = ResidualBlock(depth * 4, depth * 2)
        self.upsample4 = UpsampleBlock(depth * 2, depth)
        self.residual9 = ResidualBlock(depth * 2, depth)
        self.convblock = ConvolutionalBlock(depth, 1, activation_fn=nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        rs1 = self.residual1(x)
        x = self.maxpool1(rs1)
        rs2 = self.residual2(x)
        x = self.maxpool2(rs2)
        rs3 = self.residual3(x)
        x = self.maxpool3(rs3)
        rs4 = self.residual4(x)
        x = self.maxpool4(rs4)
        x = self.residual5(x)
        x = self.upsample1(x)
        x = self.residual6(torch.cat((x, rs4), dim=1))
        x = self.upsample2(x)
        x = self.residual7(torch.cat((x, rs3), dim=1))
        x = self.upsample3(x)
        x = self.residual8(torch.cat((x, rs2), dim=1))
        x = self.upsample4(x)
        x = self.residual9(torch.cat((x, rs1), dim=1))
        x = self.convblock(x)
        x = x.squeeze(1)

        return x


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(p=0.3)
        self.LSTM = nn.LSTM(128, 512, batch_first=True, bidirectional=True)
        self.LSTM2 = nn.LSTM(1024, 512, batch_first=True, bidirectional=True)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dense1 = nn.Linear(1024, 32)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.LSTM(x)
        x, _ = self.LSTM2(x)
        x = self.dropout4(x)
        x = x[:, -1, :]
        x = self.dense1(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout1 = nn.Dropout(p=0.4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout2 = nn.Dropout(p=0.4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout3 = nn.Dropout(p=0.4)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.dropout3(x)
        x = F.leaky_relu(self.conv4(x))

        return x


class LastChance(nn.Module):
    def __init__(self):
        super(LastChance, self).__init__()
        self.noise = GaussianNoise(std=0.01) 
        self.skipconv = nn.Conv1d(1, 128, kernel_size=1)
        self.skip_dropout = nn.Dropout(p=0.4)
        self.convblock1 = ConvBlock(5)
        self.convblock2 = ConvBlock(7)
        self.convblock3 = ConvBlock(9)
        self.LSTM = nn.LSTM(128 * 4, 256, num_layers=4, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.4)
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 32)

    def forward(self, x, noise=True):
        if noise:
            x = self.noise(x)
        x = x.unsqueeze(1)
        skip = F.leaky_relu(self.skipconv(x))
        skip = self.skip_dropout(skip)
        x1 = self.convblock1(x)
        x2 = self.convblock2(x)
        x3 = self.convblock3(x)
        x = torch.cat([skip, x1, x2, x3], dim=1)
        x = x.permute(0, 2, 1)
        x, _ = self.LSTM(x)
        x = self.dropout1(x)
        x = x[:, -1, :]
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x
    
class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.01):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x

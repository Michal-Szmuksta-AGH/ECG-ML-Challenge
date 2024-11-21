import os
from typing import Union

import typer

import dataset.processing as processing
import model.training as training
from model.models import LSTMModel
from config import RAW_DATA_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR

app = typer.Typer()


@app.command()
def model_summary(model_type: str = "LSTM") -> None:
    """
    Print the summary of the specified model type.

    :param model_type: Type of model to summarize.
    """
    if model_type == "LSTM":
        model = LSTMModel()
        print(model)
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


@app.command()
def download_dataset(dataset_name: str, dataset_dir: str = RAW_DATA_DIR) -> None:
    """
    Download the specified WFDB dataset.

    :param dataset_name: Name of the dataset to download.
    :param dataset_dir: Directory to save the downloaded dataset.
    """
    processing.download_wfdb_dataset(dataset_name, os.path.join(dataset_dir, dataset_name))


@app.command()
def clear_dataset(dataset_name: Union[str, None] = None, data_type: str = "both") -> None:
    """
    Clear interim or processed data for a specific dataset or all datasets.

    :param dataset_name: Name of the dataset or None to clear all datasets.
    :param data_type: Type of data to clear ('interim', 'processed', or 'both').
    """
    processing.clear_data(dataset_name, data_type)


@app.command()
def preprocess_dataset(dataset_name: str, target_fs: int, verbosity: str = "INFO") -> None:
    """
    Preprocess the specified WFDB dataset.

    :param dataset_name: Name of the dataset.
    :param target_fs: Target sampling frequency.
    :param verbosity: Verbosity level for logging.
    """
    processing.preprocess_dataset(dataset_name, target_fs, verbosity)


@app.command()
def process_dataset(
    chunk_size: int, test_size: float = 0.2, val_size: float = 0.1, verbosity: str = "INFO"
) -> None:
    """
    Process the preprocessed data.

    :param chunk_size: Number of samples per chunk.
    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param verbosity: Verbosity level for logging.
    """
    processing.process_dataset(chunk_size, test_size, val_size, verbosity)


@app.command()
def create_dataset(
    chunk_size: int,
    target_fs: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    verbosity: str = "INFO",
) -> None:
    """
    Execute the full pipeline: clear data, preprocess all datasets, and process the data.

    :param chunk_size: Number of samples per chunk.
    :param target_fs: Target sampling frequency.
    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param verbosity: Verbosity level for logging.
    """
    processing.clear_data(None, "both")

    for dataset_name in os.listdir(RAW_DATA_DIR):
        if os.path.isdir(os.path.join(RAW_DATA_DIR, dataset_name)):
            processing.preprocess_dataset(dataset_name, target_fs, verbosity)

    processing.process_dataset(chunk_size, test_size, val_size, verbosity)


@app.command()
def train_model(
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    model_type: str = "LSTM",
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
    :param val_data_dir: Directory for validation data.
    :param verbosity: Logging verbosity level.
    :param resume_model: Path to a pre-trained model to resume training. If None, train from scratch.
    """
    training.train_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_type=model_type,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        verbosity=verbosity,
        resume_model=resume_model,
    )


if __name__ == "__main__":
    app()

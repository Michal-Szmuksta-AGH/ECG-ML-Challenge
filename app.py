import os
import typer
import shutil
from typing import Union
from loguru import logger

import src.dataset.processing as processing
import src.dataset.processingv2 as processingv2
import src.model.training as training
import src.model.evaluate as evaluate
from src.model.models import get_model
from src.config import (
    RAW_DATA_DIR,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
    TEST_DATA_DIR,
    MODELS_DIR,
    MINIMAL_RUNTIME_DIR,
    SRC_DIR,
)

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def model_summary(model_type: str = "LSTMModel") -> None:
    """
    Print the summary of the specified model type.

    :param model_type: Type of model to summarize.
    """
    model = get_model(model_type)
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


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
def preprocess_dataset(
    dataset_name: str,
    target_fs: int,
    chunk_size: int,
    step: int = 5,
    verbosity: str = "INFO",
    version: int = 2,
) -> None:
    """
    Preprocess the specified WFDB dataset.

    :param dataset_name: Name of the dataset.
    :param target_fs: Target sampling frequency.
    :param chunk_size: Number of samples per chunk.
    :param step: Step of window step while creating chunks.
    :param verbosity: Verbosity level for logging.
    :param version: Version of the preprocessing function to use (1 or 2).
    """
    if version == 1:
        processing.preprocess_dataset(dataset_name, target_fs, chunk_size, verbosity)
    elif version == 2:
        processingv2.preprocess_dataset_v2(dataset_name, target_fs, chunk_size, step, verbosity)
    else:
        raise ValueError("Invalid version specified. Use 1 or 2.")


@app.command()
def process_dataset(
    test_size: float = 0.2, val_size: float = 0.1, verbosity: str = "INFO", version: int = 2
) -> None:
    """
    Process the preprocessed data.

    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param verbosity: Verbosity level for logging.
    :param version: Version of the processing function to use (1 or 2).
    """
    if version == 1:
        processing.process_dataset(test_size, val_size, verbosity)
    elif version == 2:
        processingv2.process_dataset_v2(test_size, val_size, verbosity)
    else:
        raise ValueError("Invalid version specified. Use 1 or 2.")


@app.command()
def create_dataset(
    chunk_size: int,
    target_fs: int,
    include_datasets: str,
    step: int = 64,
    test_size: float = 0.2,
    val_size: float = 0.1,
    verbosity: str = "INFO",
    version: int = 2,
) -> None:
    """
    Execute the full pipeline: clear data, preprocess specified datasets, and process the data.

    :param chunk_size: Number of samples per chunk.
    :param target_fs: Target sampling frequency.
    :param include_datasets: Space-separated list of datasets to include.
    :param step: Step of window step while creating chunks.
    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param verbosity: Verbosity level for logging.
    :param version: Version of the processing function to use (1 or 2).
    """
    processing.clear_data(None, "both")

    include_list = include_datasets.split()
    for dataset_name in include_list:
        if not os.path.isdir(os.path.join(RAW_DATA_DIR, dataset_name)):
            raise ValueError(f"Dataset {dataset_name} does not exist in {RAW_DATA_DIR}")

    for dataset_name in include_list:
        if version == 1:
            processing.preprocess_dataset(dataset_name, target_fs, chunk_size, verbosity)
        elif version == 2:
            processingv2.preprocess_dataset_v2(
                dataset_name, target_fs, chunk_size, step, verbosity
            )
        else:
            raise ValueError("Invalid version specified. Use 1 or 2.")

    if version == 1:
        processing.process_dataset(test_size, val_size, verbosity)
    elif version == 2:
        processingv2.process_dataset_v2(test_size, val_size, verbosity)
    else:
        raise ValueError("Invalid version specified. Use 1 or 2.")


@app.command()
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


@app.command()
def package_runtime(
    model_file: str, model_type: str, trained_chunk_size: int, trained_fs: int
) -> None:
    """
    Package the minimal runtime of the project into a zip file.

    :param model_file: Name of the model file to include.
    :param model_type: Type of the model.
    :param trained_chunk_size: Chunk size used during training.
    :param trained_fs: Sampling frequency used during training.
    """
    logger.info(f'Packaging minimal runtime with model file "{model_file}"')

    os.makedirs(MINIMAL_RUNTIME_DIR, exist_ok=True)

    logger.info("Clearing existing model files")
    existing_model_files = os.listdir(MINIMAL_RUNTIME_DIR)
    existing_model_files = [
        f for f in existing_model_files if f.endswith(".pth") or f.endswith(".pt")
    ]
    for existing_model_file in existing_model_files:
        os.remove(os.path.join(MINIMAL_RUNTIME_DIR, existing_model_file))
    logger.info(f"Cleared {len(existing_model_files)} existing model files")

    model_src = os.path.join(model_file)
    model_dst = os.path.join(MINIMAL_RUNTIME_DIR, model_file.split("/")[-1])
    logger.info(f"Copying model file from {model_src} to {model_dst}")
    shutil.copy(model_src, model_dst)

    model_class_src = os.path.join(SRC_DIR, "model", "models.py")
    model_class_dst = os.path.join(MINIMAL_RUNTIME_DIR, "models.py")
    logger.info(f"Copying models.py from {model_class_src} to {model_class_dst}")
    shutil.copy(model_class_src, model_class_dst)

    logger.info("Creating config.py")
    config_path = os.path.join(MINIMAL_RUNTIME_DIR, "config.py")
    if os.path.exists(config_path):
        os.remove(config_path)
    with open(config_path, "w") as config_file:
        config_file.write(f"TRAINED_MODEL_TYPE = '{model_type}'\n")
        config_file.write(f"TRAINED_CHUNK_SIZE = {trained_chunk_size}\n")
        config_file.write(f"TRAINED_FS = {trained_fs}\n")
        config_file.write(f"MODEL_FILE = '{model_file.split('/')[-1]}'\n")
        config_file.write(f"DEST_DIR = './'\n")

    logger.info(f"Creating zip file: {MINIMAL_RUNTIME_DIR}.zip")
    shutil.make_archive(MINIMAL_RUNTIME_DIR, "zip", MINIMAL_RUNTIME_DIR)


@app.command()
def evaluate_model(
    model_type: str,
    state_dict_name: str,
    models_dataset_dir: str = MODELS_DIR,
    num_samples: int = 1,
) -> None:
    """
    Evaluate the trained model on a specified number of random samples from the test dataset.

    :param model_type: Type of the model.
    :param state_dict_name: Name of the state_dict file.
    :param models_dataset_dir: Directory of the models dataset.
    :param num_samples: Number of samples to evaluate.
    """
    evaluate.evaluate_model(
        model_type, state_dict_name, TEST_DATA_DIR, models_dataset_dir, num_samples
    )


@app.command()
def evaluate_model_v2(
    model_type: str,
    state_dict_name: str,
    model_chunk_size: int,
    model_fs: int,
    database: str,
    record: str,
    models_dataset_dir: str = MODELS_DIR,
    xlim_min: int = None,  # New argument
    xlim_max: int = None,  # New argument
) -> None:
    """
    Evaluate the trained model on a specified record from the database.

    :param model_type: Type of the model.
    :param state_dict_name: Name of the state_dict file.
    :param model_chunk_size: Chunk size used during training.
    :param model_fs: Sampling frequency used during training.
    :param models_dataset_dir: Directory of the models dataset.
    :param database: Name of the database to evaluate.
    :param record: Name of the record to evaluate.
    :param xlim_min: Minimum x-axis limit for the plots.
    :param xlim_max: Maximum x-axis limit for the plots.
    """
    evaluate.evaluate_model_v2(
        model_type,
        state_dict_name,
        model_chunk_size,
        model_fs,
        models_dataset_dir,
        database,
        record,
        xlim_min,  # Pass new argument
        xlim_max,  # Pass new argument
    )


@app.command()
def evaluate_tensor_shapes(
    model_type: str,
    batch_size: int = 8,
    models_dataset_dir: str = MODELS_DIR,
) -> None:
    """
    Evaluate tensor shapes during model processing for a random batch from the training dataset.

    :param model_type: Type of the model.
    :param batch_size: Size of the batch to evaluate.
    :param models_dataset_dir: Directory of the models dataset.
    """
    evaluate.evaluate_tensor_shapes(model_type, batch_size, models_dataset_dir)


if __name__ == "__main__":
    app()

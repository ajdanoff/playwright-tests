import torch
import pytest
from ml.train_model import MLModel  # Import your MLModel class

@pytest.fixture(scope="module")
def setup_ml_model():
    # Setup MLModel with small dataset for tests, adjust paths and params accordingly
    """
    Pytest fixture that initializes the MLModel instance with test configuration.

    This fixture sets up the MLModel with a small batch size, short epochs,
    and uses sample datasets for testing. It is scoped to module so setup runs once
    per test module session and can be reused in multiple tests.

    Returns:
        MLModel: Initialized MLModel instance ready for testing.
    """
    model = MLModel(
        annotations="annotations.csv",
        data_folder="./test_data",
        query=["spiders", "butterflies"],
        validation_split=0.2,
        batch_size=8,
        epochs=1,
        shuffle_dataset=False,
        learning_rate=1e-3
    )
    return model

def test_model_initialization(setup_ml_model):
    """
    Test that the MLModel creates a neural network model on the correct device.

    Verifies that the model attribute is a PyTorch nn.Module and that the device
    is either CPU or CUDA GPU.

    Args:
        setup_ml_model (MLModel): Fixture providing initialized model.
    """
    model = setup_ml_model
    assert model.model is not None
    assert isinstance(model.model, torch.nn.Module)
    assert model.device.type in ["cpu", "cuda"]

def test_data_and_model_shapes(setup_ml_model):
    """
    Test that input data batch shapes match model expectations.

    Checks that input tensors have 4 dimensions with 3 channels (RGB),
    and that the flattened input size corresponds to the model's input layer size.
    Also checks that the model output shape matches batch size and number of output classes.

    Args:
        setup_ml_model (MLModel): Fixture providing initialized model.
    """
    model = setup_ml_model
    dataloader = model.train_dataloader
    X, y = next(iter(dataloader))
    assert X.ndim == 4  # (batch_size, C, H, W)
    assert X.shape[1] == 3  # RGB channels
    x_flatten = X.view(X.size(0), -1)
    expected_features = 3 * model.img_size[0] * model.img_size[1]
    assert x_flatten.shape[1] == expected_features
    pred = model.model(X.to(model.device))
    assert pred.shape == (X.size(0), len(model.int_labels))

def test_single_train_step(setup_ml_model):
    """
    Test that a single training step runs forward and backward without errors.

    Executes one batch from train dataloader:
    - Forward pass
    - Loss computation
    - Backward pass
    - Optimizer step
    - Zero grads
    Asserts that output shapes are consistent and loss is positive.

    Args:
        setup_ml_model (MLModel): Fixture providing initialized model.
    """
    model = setup_ml_model
    model.model.train()
    dataloader = model.train_dataloader
    X, y = next(iter(dataloader))
    print(f"Input batch shape: {X.shape}")  # Expect (batch_size, 3, H, W)
    X, y = X.to(model.device), y.to(model.device)
    pred = model.model(X)
    print(f"Output batch shape: {pred.shape}")  # Expect (batch_size, num_classes)
    assert pred.shape[0] == X.shape[0]
    loss = model.loss_fn(pred, y)
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad()
    assert loss.item() > 0

def test_inference_output_shape(setup_ml_model):
    """
    Test inference to confirm output shapes and device alignment on test data.

    Runs a batch through the model in evaluation mode without gradients.
    Asserts output batch size and output layer dimensions.

    Args:
        setup_ml_model (MLModel): Fixture providing initialized model.
    """
    model = setup_ml_model
    model.model.eval()
    with torch.no_grad():
        X, _ = next(iter(model.test_dataloader))
        X = X.to(model.device)
        outputs = model.model(X)
        assert outputs.shape[0] == X.shape[0]
        assert outputs.shape[1] == len(model.int_labels)

import os

def test_train_epochs_runs(setup_ml_model):
    """
    Test that training for multiple epochs completes without exceptions,
    saves model to default folder, and the saved file exists on disk.

    Args:
        setup_ml_model (MLModel): Pytest fixture that provides an initialized MLModel.
    """
    model = setup_ml_model
    model.epochs = 5  # Limit epochs for test speed
    model.train_epochs()  # Run training loop
    model.save_model()    # Save model to disk using default path
    saved_path = model.model_path()

    # Assert that the model file exists after saving
    assert os.path.exists(saved_path), f"Model file not found after saving: {saved_path}"

    # Optional: assert non-empty file size
    assert os.path.getsize(saved_path) > 0, "Saved model file is empty"


def test_load_model_eval(setup_ml_model):
    """
    Test loading the saved model from disk without exceptions,
    then run the test loop to evaluate model on validation data,
    verifying that accuracy and loss metrics are reasonable.

    Args:
        setup_ml_model (MLModel): Pytest fixture that provides an initialized MLModel.
    """
    model = setup_ml_model
    model.load_model(model.model_path())  # Load saved model

    # Run evaluation and capture output by temporarily redirecting stdout
    import io
    import sys

    captured_output = io.StringIO()
    sys.stdout = captured_output
    model.test_loop()  # Execute evaluation loop
    sys.stdout = sys.__stdout__

    output = captured_output.getvalue()

    # Check output includes accuracy % and loss
    assert "Accuracy" in output, "Evaluation output missing 'Accuracy'"
    assert "Avg loss" in output, "Evaluation output missing 'Avg loss'"

    # Optional: parse accuracy and check it's between 0 and 100
    import re
    acc_match = re.search(r"Accuracy:\s*([\d\.]+)%", output)
    if acc_match:
        accuracy = float(acc_match.group(1))
        assert 50.0 <= accuracy <= 100.0, f"Accuracy {accuracy} out of valid range"


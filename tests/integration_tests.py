import sys
import pytest
from unittest.mock import patch, AsyncMock

import main  # Your main.py module


@pytest.mark.asyncio
@patch("main.async_playwright")
async def test_crawl_mode(mock_playwright):
    """
    Test running the CLI in crawl mode (mode=0).
    Verify ImagesCrawler .crawl() is called with expected args.
    """
    mock_playwright_instance = AsyncMock()
    mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()

    mock_playwright_instance.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    with patch("crawler.crawler.ImagesCrawler.crawl", new_callable=AsyncMock) as mock_crawl:
        test_argv = [
            "main.py",
            "--mode", "0",
            "--host", "bing.com",
            "--query", "spiders",
            "--label", "spider",
            "--nscrolls", "2",
            "--ddir", "./testdata",
            "--minimsize", "5000",
        ]
        with patch.object(sys, "argv", test_argv):
            await main.main()

        mock_crawl.assert_called_once()
        kwargs = mock_crawl.call_args.kwargs
        assert kwargs["host"] == "bing.com"
        assert kwargs["query"] == "spiders"
        assert kwargs["label"] == "spider"
        assert kwargs["num_scrolls"] == 2
        assert kwargs["data_dir"] == "./testdata"
        assert kwargs["min_imsize"] == 5000


def test_parse_args_train_mode():
    """
    Test argument parsing for training mode with model_out_fldr.
    """
    test_argv = [
        "main.py",
        "--mode", "1",
        "-ann", "ann.csv",
        "--ddir", "./data",
        "--query", "cats",
        "--epochs", "2",
        "--batch-size", "16",
        "--learning-rate", "0.01",
        "--validation-split", "0.25",
        "--model-out-fldr", "./my_model_out"
    ]
    with patch.object(sys, "argv", test_argv):
        args = main.parse_arguments()

    assert args.mode == 1
    assert args.annotations == "ann.csv"
    assert args.ddir == "./data"
    assert args.query == "cats"
    assert args.epochs == 2
    assert args.batch_size == 16
    assert pytest.approx(args.learning_rate, 1e-6) == 0.01
    assert pytest.approx(args.validation_split, 1e-6) == 0.25
    assert args.model_out_fldr == "./my_model_out"


def init_side_effect(self, *args, **kwargs):
    """
    Mock __init__ to set mod_out_fldr attribute to prevent errors in tests.
    """
    self.query = kwargs.get('query', 'test')
    self.mod_out_fldr = kwargs.get('mod_out_fldr', './model')
    # You can set other expected attributes here if needed
    return None


@patch("ml.train_model.MLModel.__init__", new=init_side_effect)
@patch("ml.train_model.MLModel.train_epochs")
@patch("ml.train_model.MLModel.save_model")
def test_train_mode_calls(mock_save, mock_train):
    """
    Test that ML model training and save_model called in mode 1 with model_out_fldr.
    """
    test_argv = [
        "main.py", "--mode", "1",
        "--model-out-fldr", "./my_model_out"
    ]
    with patch.object(sys, "argv", test_argv):
        import asyncio
        asyncio.run(main.main())

    mock_train.assert_called_once()
    mock_save.assert_called_once()


@patch("ml.train_model.MLModel.test_loop")
@patch("ml.train_model.MLModel.load_model")
@patch("ml.train_model.MLModel.__init__", new=init_side_effect)
def test_test_mode_calls(mock_load, mock_test):
    """
    Test that MLModel load_model and test_loop are called in mode 2 with model_out_fldr.
    """
    test_argv = [
        "main.py", "--mode", "2",
        "--model-out-fldr", "./my_model_out"
    ]
    with patch.object(sys, "argv", test_argv):
        import asyncio
        asyncio.run(main.main())

    mock_load.assert_called_once()
    mock_test.assert_called_once()

import asyncio
from playwright.async_api import async_playwright, Playwright
import argparse

from crawler.crawler import ImagesCrawler


def parse_arguments():
    """
    Setup argparse.ArgumentParser and parse CLI arguments including crawler and ML model parameters.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog='MagicFlute',
        description='Image crawler and ML model trainer using Playwright and PyTorch',
        epilog='The Song of the Bell',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Crawler parameters
    parser.add_argument('-hs', '--host', type=str, default='bing.com', help='Host to crawl')
    parser.add_argument('-qr', '--query', type=str, default='spiders', help='Search query')
    parser.add_argument('-lbl', '--label', type=str, default='spider', help='Label for images')
    parser.add_argument('-ns', '--nscrolls', type=int, default=5, help='Number of frame scrolls')
    parser.add_argument('-dd', '--ddir', type=str, default='./data', help='Directory to save images')
    parser.add_argument('-mnims', '--minimsize', type=int, default=8000, help='Minimum image size to save in bytes')

    # ML model parameters
    parser.add_argument('-ann', '--annotations', type=str, default='annotations.csv', help='Filename of annotations CSV')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--validation-split', type=float, default=0.3, help='Fraction of data used for validation (0-1)')
    parser.add_argument('--model-out-fldr', type=str, default='./model', help='Output folder for saved models')

    # Mode selector (0=crawl, 1=train, 2=test)
    parser.add_argument(
        '--mode', type=int, choices=[0, 1, 2], default=0,
        help='Mode selector: 0 - crawl, 1 - train model, 2 - test model'
    )

    return parser.parse_args()


async def run_crawler(playwright: Playwright, **kwargs):
    """
    Launch Playwright browser context and run the image crawler.

    Args:
        playwright (Playwright): Playwright async context.
        kwargs: Keyword arguments for the crawler configuration.
    """
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()

    img_crawler = ImagesCrawler(playwright, page, kwargs.get('host'), kwargs.get('data_dir'))
    await img_crawler.crawl(**kwargs)
    img_crawler.write_annotations(kwargs.get('query'), kwargs.get('annotations'))

    await context.close()
    await browser.close()


async def main() -> None:
    """
    Main entry point: parse arguments and execute selected mode.

    Modes:
        0 - Crawl images asynchronously.
        1 - Train ML model synchronously.
        2 - Test ML model synchronously.
    """
    args = parse_arguments()

    if args.mode == 0:
        # Gather crawler configuration kwargs
        crawl_kwargs = {
            'host': args.host,
            'query': args.query,
            'label': args.label,
            'data_dir': args.ddir,
            'num_scrolls': args.nscrolls,
            'min_imsize': args.minimsize,
            'annotations': args.annotations
        }
        async with async_playwright() as playwright:
            await run_crawler(playwright, **crawl_kwargs)

    elif args.mode == 1:
        # ML model training mode
        from ml.train_model import MLModel
        model = MLModel(
            annotations=args.annotations,
            data_folder=args.ddir,
            query=args.query,
            validation_split=args.validation_split,
            shuffle_dataset=True,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_out_fldr=args.model_out_fldr,
        )
        model.train_epochs()
        model.save_model()

    elif args.mode == 2:
        # ML model testing mode
        from ml.train_model import MLModel
        model = MLModel(
            annotations=args.annotations,
            data_folder=args.ddir,
            query=args.query,
            validation_split=args.validation_split,
            shuffle_dataset=True,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_out_fldr=args.model_out_fldr,
        )
        model.load_model(model.model_path())
        model.test_loop()


if __name__ == "__main__":
    # Run main async function in event loop
    asyncio.run(main())

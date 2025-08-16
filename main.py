import asyncio

from playwright.async_api import async_playwright, Playwright
import argparse

from crawler.crawler import ImagesCrawler

parser = argparse.ArgumentParser(
        prog='MagicFlute',
        description='A simple image crawler written in python using PlayWright',
        epilog='The Song of the Bell'
    )
parser.add_argument('-hs', '--host', type=str, default='https://bing.com', help='a host to crawl')
parser.add_argument('-qr', '--query', type=str, default='spiders', help='a query to search for')
parser.add_argument('-ns', '--nscrolls', type=int, default=5, help='number of frame scrolls')
parser.add_argument('-dd', '--ddir', type=str, default='./data', help='directory to save images')


async def run(playwright: Playwright, **kwargs):
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()
    img_crawler = ImagesCrawler(playwright, page, kwargs.get('host', None), kwargs.get('data_dir', None))
    await img_crawler.crawl(**kwargs)
    await context.close()
    await browser.close()

async def main() -> None:
    args = parser.parse_args()
    kwargs = {'host': args.host, 'query': args.query, 'data_dir': args.ddir, 'num_scrolls': args.nscrolls}

    async with async_playwright() as playwright:
        await run(playwright, **kwargs)

if __name__ == "__main__":
    asyncio.run(main())

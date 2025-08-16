import pytest
import asyncio
from playwright.async_api import async_playwright
from crawler.crawler import ImagesCrawler  # Replace with your actual import path

@pytest.mark.asyncio
async def test_crawl_basic():
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)  # headed mode enabled
        context = await browser.new_context()
        page = await context.new_page()

        host = "https://bing.com"
        data_folder = "./test_data"
        crawler = ImagesCrawler(playwright, page, host, data_folder)

        # Run crawl with limited scrolls and query
        await crawler.crawl(query="test", num_scrolls=1)

        import os
        assert os.path.exists(data_folder)
        assert any(os.scandir(data_folder))  # check at least one file/folder inside

        await context.close()
        await browser.close()

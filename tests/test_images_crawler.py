import pytest
import asyncio
from playwright.async_api import async_playwright
from crawler.crawler import ImagesCrawler  # Replace with your actual import path

@pytest.mark.parametrize("host, data_folder, query, num_scrolls", [
    ("bing.com", "./test_data", "spiders", 1)
])
@pytest.mark.asyncio
async def test_crawl_basic(host, data_folder, query, num_scrolls):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)  # headed mode enabled
        context = await browser.new_context()
        page = await context.new_page()

        crawler = ImagesCrawler(playwright, page, host, data_folder)

        # Run crawl with limited scrolls and query
        await crawler.crawl(query=query, num_scrolls=num_scrolls)

        import os
        assert os.path.exists(data_folder)
        assert any(os.scandir(data_folder))  # check at least one file/folder inside

        await context.close()
        await browser.close()

import asyncio
import copy
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from playwright.async_api import Page, Playwright

from crawler.html_parser import ImgHTMLParser


class CrawlerType(str, Enum):
    IMAGES_CRAWLER = "Images"


MY_IMAGE_CRAWLER = "MyImageCrawler"
PAGE_DOWN_JS = """
div = document.querySelector('#b_content');
div.scrollBy(0, window.innerHeight);
"""
IMAGE_FORMATS = ["webp", "png", "jpeg", "tiff", "webp"]

class Crawler(ABC):

    _cr_type: CrawlerType

    def __init__(self, cr_type: CrawlerType):
        self._cr_type = cr_type

    @property
    def cr_type(self):
        return self._cr_type

    @abstractmethod
    async def crawl(self, **kwargs):
        raise NotImplementedError("'crawl' is not implemented !")


class ImagesCrawler(Crawler):

    _page: Page
    _playwright: Playwright
    _host: str
    _data_folder: str
    _robot_parser: RobotFileParser
    _crawl_delay: float = 0.0

    def __init__(self, playwright: Playwright, page: Page, host, data_folder):
        Crawler.__init__(self, CrawlerType.IMAGES_CRAWLER)
        self._playwright = playwright
        self._page = page
        self.playwright.selectors.set_test_id_attribute('data-bm')
        self.ihp = ImgHTMLParser()
        self._host = host
        self._data_folder = data_folder

        self._robot_parser = RobotFileParser()
        robots_url = self._get_robots_url(host)
        self._robot_parser.set_url(robots_url)
        self._robot_parser.read()

        # Get crawl-delay for your user-agent (default to 0)
        user_agent = MY_IMAGE_CRAWLER
        delay = self._robot_parser.crawl_delay(user_agent)
        self._crawl_delay = delay if delay is not None else 0.0

    @property
    def host(self):
        return self._host

    @property
    def data_folder(self):
        return self._data_folder

    @property
    def page(self):
        return self._page

    @property
    def playwright(self):
        return self._playwright

    @staticmethod
    def _get_robots_url(site_url: str) -> str:
        parsed_url = urlparse(site_url)
        return f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

    def _can_crawl(self, url: str) -> bool:
        user_agent = MY_IMAGE_CRAWLER
        return self._robot_parser.can_fetch(user_agent, url)

    @staticmethod
    def is_image(cont_type):
        img = cont_type.split("/")
        if len(img) == 2 and (img[1] in IMAGE_FORMATS):
            return img[1]
        return None

    def request_download(self, src, meta, folder):
        folder_path = f"{self.data_folder}/{folder}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            logging.getLogger().info("directory %s created", folder_path)
        else:
            logging.getLogger().info("directory %s already exists", folder_path)

        response = requests.get(src)
        if response.status_code == 200:
            print(response.headers)
            ext = self.is_image(response.headers['Content-Type'])
            if ext and int(response.headers['Content-Length']) > 8000:
                safe_name = hashlib.md5(str(meta).encode()).hexdigest()
                loc_file = f"{folder_path}/{safe_name}.{ext}"
                with open(loc_file, 'wb') as f:
                    f.write(response.content)
                logging.getLogger().info("file %s downloaded successfully", loc_file)
                return loc_file
        else:
            logging.getLogger().info("Failed to download file. Status code: %d", response.status_code)
        return None

    async def crawl(self, **kwargs):
        num_scrolls, query = await self.extract_args(kwargs)

        if not self._can_crawl(self._host):
            logging.info(f"Crawling disallowed by robots.txt on {self._host}")
            return

        await self.page.goto(self.host)
        await self.page.get_by_role("searchbox", name="Enter your search term").click()
        await self.page.get_by_role("searchbox", name="Enter your search term").fill(f"{query} {self.cr_type.value}")
        await self.page.get_by_role("searchbox", name="Enter your search term").press("Enter")

        async with self.page.expect_popup() as page1_info:
            await self.page.get_by_label("Search Filter").get_by_role("link", name=self.cr_type.value).click()
        page1 = await page1_info.value
        await page1.wait_for_selector('#b_content')
        await asyncio.sleep(5)

        prev_content = await page1.content()
        await self.parse_download(page1, prev_content, query)

        for i in range(num_scrolls):
            await page1.keyboard.press("PageDown", delay=1000)
            await page1.wait_for_timeout(5000)

            await self._respect_crawl_delay()

            self.ihp.srcs.clear()
            content = await page1.content()
            assert content != prev_content
            await self.parse_download(page1, prev_content, query)
        # await page1.screenshot(path=f"{query}.png")

    async def _respect_crawl_delay(self):
        if self._crawl_delay > 0:
            logging.info(f"Respecting crawl-delay of {self._crawl_delay} seconds")
            await asyncio.sleep(self._crawl_delay)


    async def parse_download(self, page1, content, query):
        self.ihp.srcs.clear()
        self.ihp.feed(content)
        srcs = copy.deepcopy(self.ihp.srcs)
        for src, meta in srcs.items():
            if self.request_download(src, meta, f"{query}"):
                if 'data-bm' in meta:
                    # lihp = ImgHTMLParser()
                    try:
                        await page1.get_by_test_id(meta['data-bm']).click()
                        await asyncio.sleep(1)
                        l_content = await page1.content()
                        self.ihp.srcs.clear()
                        self.ihp.feed(l_content)
                        if self.ihp.srcs:
                            for lsrc, lmeta in self.ihp.srcs.items():
                                self.request_download(lsrc, lmeta, f"{query}")
                        await page1.locator("iframe[id=\"OverlayIFrame\"]").content_frame.get_by_label(
                            "Close image").click()
                        await asyncio.sleep(1)
                    except TimeoutError as te:
                        logging.getLogger().info("timeout error: %s", te)

    @staticmethod
    async def extract_args(kwargs):
        query = kwargs.get("query", "butterfly")
        num_scrolls = kwargs.get("num_scrolls", 5)
        return num_scrolls, query

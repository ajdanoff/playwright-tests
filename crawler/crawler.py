import asyncio
import copy
import csv
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser


import requests
from playwright.async_api import Page, Playwright, TimeoutError

from crawler.admin import check_folder_mk, hash_file_name
from crawler.html_parser import ImgHTMLParser
from crawler.search_services import BingService, get_service_by_host


class CrawlerType(str, Enum):
    """Enum to define crawler types. Currently only ImagesCrawler defined."""
    IMAGES_CRAWLER = "Images"


MY_IMAGE_CRAWLER = "MyImageCrawler"
PAGE_DOWN_JS = """
div = document.querySelector('#b_content');
div.scrollBy(0, window.innerHeight);
"""
IMAGE_FORMATS = ["webp", "png", "jpeg", "tiff", "webp"]


class Crawler(ABC):
    """
    Abstract base class for all crawlers.

    Defines the required interface and common properties for crawlers.
    """
    _cr_type: CrawlerType
    _annotations: list[dict]

    def __init__(self, cr_type: CrawlerType):
        self._cr_type = cr_type
        self._annotations = []

    @property
    def cr_type(self):
        """Returns the type of the crawler instance."""
        return self._cr_type

    @property
    def annotations(self):
        """Returns the list of current annotations collected."""
        return self._annotations

    @abstractmethod
    async def crawl(self, **kwargs):
        """Abstract asynchronous crawl method to be implemented by subclasses."""
        raise NotImplementedError("'crawl' is not implemented !")

    @abstractmethod
    def write_annotations(self, query, csv_file: str):
        """Abstract method for persisting annotations to file."""
        raise NotImplementedError("'write_annotations' is not implemented !")

    @abstractmethod
    def make_annotation(self, **kwargs):
        """Abstract method to create an annotation entry."""
        raise NotImplementedError("'make_annotations' is not implemented !")


class ImagesCrawler(Crawler):
    """
    Concrete implementation of an images crawler.

    Uses Playwright for navigation, BingService for search service interactions,
    and extracts images with relevant metadata, respecting robots.txt crawling rules.
    """
    _page: Page
    _playwright: Playwright
    _host: str
    _data_folder: str
    _robot_parser: RobotFileParser
    _crawl_delay: float = 0.0
    _search_service: BingService
    _fieldnames: list[str]

    def __init__(self, playwright: Playwright, page: Page, host, data_folder):
        """
        Initialize the ImagesCrawler.

        Args:
            playwright (Playwright): Playwright instance.
            page (Page): Playwright page to automate browser actions.
            host (str): The host/URL to crawl.
            data_folder (str): Base folder to save crawled images and annotations.
        """
        Crawler.__init__(self, CrawlerType.IMAGES_CRAWLER)
        self._playwright = playwright
        self._page = page
        self._search_service = get_service_by_host(page, host)

        self.playwright.selectors.set_test_id_attribute(self.search_service.test_id_attr)
        self.ihp = ImgHTMLParser()
        self._host = host
        self._data_folder = data_folder

        self._robot_parser = RobotFileParser()
        robots_url = self._get_robots_url(self.search_service.search_url())
        self._robot_parser.set_url(robots_url)
        self._robot_parser.read()

        # Get crawl-delay for your user-agent (default to 0)
        user_agent = MY_IMAGE_CRAWLER
        delay = self._robot_parser.crawl_delay(user_agent)
        self._crawl_delay = delay if delay is not None else 0.0

        # annotations
        self._fieldnames = ["filename", "height", "label", "width"]
        # downloaded images
        self._downloaded = []

    @property
    def host(self):
        """Returns crawl target host."""
        return self._host

    @property
    def data_folder(self):
        """Returns base folder where data is stored."""
        return self._data_folder

    @property
    def page(self):
        """Returns Playwright Page object."""
        return self._page

    @property
    def playwright(self):
        """Returns Playwright instance."""
        return self._playwright

    @property
    def search_service(self):
        """Returns the configured search service handler."""
        return self._search_service

    @property
    def fieldnames(self):
        """Returns the CSV annotation field names."""
        return self._fieldnames

    @property
    def downloaded(self):

        return self._downloaded

    @staticmethod
    def _get_robots_url(site_url: str) -> str:
        """Returns list of downloaded image URLs."""
        parsed_url = urlparse(site_url)
        return f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

    def _can_crawl(self, url: str) -> bool:
        """
        Check if the crawler is allowed to crawl given URL per robots.txt.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if allowed, False otherwise.
        """
        user_agent = MY_IMAGE_CRAWLER
        return self._robot_parser.can_fetch(user_agent, url)

    @staticmethod
    def is_image(cont_type):
        """
        Check if content-type header corresponds to known image format.

        Args:
            cont_type (str): The content-type header value.

        Returns:
            str or None: Image extension if supported, else None.
        """
        img = cont_type.split("/")
        if len(img) == 2 and (img[1] in IMAGE_FORMATS):
            return img[1]
        return None

    def write_annotations(self, query, csv_file: str):
        """
        Write collected annotations to a CSV file.

        Args:
            query (str): Query term/folder name.
            csv_file (str): Name of the CSV file to write.
        """
        with open(os.path.join(self.query_folder(query), csv_file), mode="w", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.annotations)

    def make_annotation(self, **kwargs):
        """
        Create an annotation dict from fields and append to annotations list.

        Accepts keyword args matching annotation field names.
        """
        annotation = {}
        for field in self.fieldnames:
            annotation[field] = kwargs.get(field, "")
        self.annotations.append(annotation)

    def query_folder(self, query):
        """
        Return filesystem folder path for given query inside base data folder.

        Args:
            query (str): Query folder name.

        Returns:
            str: Full path.
        """
        return os.path.join(self.data_folder, query)

    def request_download(self, src, meta, query, label, min_im_size=8000):
        """
        Request download of image at src URL if conditions satisfy.

        Args:
            src (str): URL of image.
            meta (dict): Metadata dictionary containing image info.
            query (str): Query term/folder to save image to.
            label (str): Label to record for annotation.
            min_im_size (int): Minimum image size threshold in bytes.

        Returns:
            str or None: Local filepath where file was saved or None on failure.
        """
        folder_path = self.query_folder(query)
        check_folder_mk(folder_path)

        if src not in self.downloaded:
            response = requests.get(src)
            if response.status_code == 200:
                # print(response.headers)
                self.downloaded.append(src)
                ext = self.is_image(response.headers['Content-Type'])
                if ext and int(response.headers['Content-Length']) > min_im_size:
                    safe_name = hash_file_name(ext, meta)
                    self.make_annotation(filename=safe_name, height=meta.get('height', ''), label=label, width=meta.get('width', ''))
                    loc_file = os.path.join(folder_path, safe_name)
                    with open(loc_file, 'wb') as f:
                        f.write(response.content)
                    logging.getLogger().info("file %s downloaded successfully", loc_file)
                    return loc_file
            else:
                logging.getLogger().info("Failed to download file. Status code: %d", response.status_code)
        return None

    async def crawl(self, **kwargs):
        """
        Main asynchronous crawl method to extract images.

        Args:
            kwargs: Optional arguments - query, label, num_scrolls, min_im_size
        """
        num_scrolls, query, label, min_im_size = await self.extract_args(kwargs)

        if not self._can_crawl(self._host):
            logging.info(f"Crawling disallowed by robots.txt on {self._host}")
            return

        await self.search_service.navigate_host()
        await self.search_service.search_query(f"{query} {self.cr_type.value}")
        page1 = await self.search_service.click_obj_filter(self.cr_type.value)

        await asyncio.sleep(5)

        prev_content = await page1.content()
        await self.parse_download(page1, prev_content, query, label, min_im_size)

        for i in range(num_scrolls):
            await asyncio.sleep(10)

            await page1.keyboard.press("PageDown", delay=1000)
            await page1.wait_for_timeout(5000)

            await self._respect_crawl_delay()

            self.ihp.srcs.clear()
            content = await page1.content()
            assert content != prev_content
            await self.parse_download(page1, prev_content, query, label, min_im_size)
        # await page1.screenshot(path=f"{query}.png")

    async def _respect_crawl_delay(self):
        """Sleep for crawl delay specified in robots.txt to respect site policy."""
        if self._crawl_delay > 0:
            logging.info(f"Respecting crawl-delay of {self._crawl_delay} seconds")
            await asyncio.sleep(self._crawl_delay)

    async def parse_download(self, page1, content, query, label, min_im_size=8000):
        """
        Parse page content for image sources and download images found.

        Args:
            page1 (Page): Playwright page context.
            content (str): HTML content of the page.
            query (str): Query/folder label for saving images.
            label (str): Label to apply to downloaded images.
            min_im_size (int): Minimum file size in bytes to save.
        """
        self.ihp.srcs.clear()
        self.ihp.feed(content)
        srcs = copy.deepcopy(self.ihp.srcs)
        for src, meta in srcs.items():
            if self.request_download(src, meta, query, label, min_im_size):
                if self.search_service.test_id_attr in meta:
                    try:
                        await page1.get_by_test_id(meta[self.search_service.test_id_attr]).click(timeout=1000)
                        await asyncio.sleep(1)
                        l_content = await page1.content()
                        self.ihp.srcs.clear()
                        self.ihp.feed(l_content)
                        if self.ihp.srcs:
                            for lsrc, lmeta in self.ihp.srcs.items():
                                self.request_download(lsrc, lmeta, query, label, min_im_size)
                        await self.search_service.click_frame_close()
                        await asyncio.sleep(1)
                    except TimeoutError as te:
                        logging.getLogger().info("timeout error: %s", te)

    @staticmethod
    async def extract_args(kwargs):
        """
        Extract crawl arguments from kwargs with defaults.

        Returns:
            tuple: (num_scrolls, query, label, min_im_size)
        """
        query = kwargs.get("query", "spiders")
        label = kwargs.get("label", "spider")
        num_scrolls = kwargs.get("num_scrolls", 5)
        min_im_size = kwargs.get("min_imsize", 8000)
        return num_scrolls, query, label, min_im_size

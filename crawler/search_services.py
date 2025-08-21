import logging
from abc import ABC, abstractmethod
from enum import Enum

from playwright.async_api import Page


class SearchServices(str, Enum):
    BING = "bing.com"


def get_service_by_host(page: Page, host: str):
    domain = host.split('.')[0]
    match domain:
        case 'bing':
            return BingService(page)
        case 'google':
            return GoogleService(page, host)
        case _:
            return GoogleService(page, "google.com")


class SearchService(ABC):

    _host: str
    _page: Page
    _schema: str
    _test_id_attr: str
    _visited: list[Page]

    def __init__(self, page: Page, host: str, schema: str = "https", test_id_attr: str = ""):
        self._host = host
        self._page = page
        self._schema = schema
        self._test_id_attr = test_id_attr
        self._visited = []

    @property
    def host(self):
        return self._host

    @property
    def page(self):
        return self._page

    @property
    def schema(self):
        return self._schema

    @property
    def test_id_attr(self):
        return self._test_id_attr

    @property
    def visited(self):
        return self._visited

    async def navigate_host(self):
        await self.page.goto(self.search_url())

    @abstractmethod
    async def search_query(self, query):
        raise NotImplementedError("'search_query' is not implemented !")

    @abstractmethod
    async def click_obj_filter(self, search_obj_type: str) -> Page:
        raise NotImplementedError("'click_obj_filter' is not implemented !")

    @abstractmethod
    async def click_frame_close(self):
        raise NotImplementedError("'click_frame_close' is not implemented !")

    def search_url(self):
        return f"{self.schema}://{self.host}"


class GoogleService(SearchService):

    def __init__(self, page: Page, host: str):
        super().__init__(page, host, schema="https", test_id_attr='data-bm')

    async def search_query(self, query: str):
        await self.page.wait_for_selector("textarea[name='q']", timeout=30000)
        await self.page.fill("textarea[name='q']", query)
        await self.page.press("textarea[name='q']", "Enter")

    async def click_obj_filter(self, search_obj_type: str):
        try:
            async with self.page.expect_popup() as page1_info:
                await self.page.locator(
                    f'//nav[@class="b_scopebar"]/ul/li[@id="b-scopeListItem-{search_obj_type.lower()}"]/a').click()
            page1 = await page1_info.value
            await page1.wait_for_selector('#b_content')
            self._visited.append(page1)
            return page1
        except Exception as e:
            logging.error(f"Error clicking {search_obj_type} filter link: {e}")
            await self.page.screenshot(path=f"error_{search_obj_type}_filter_click.png")
            content = await self.page.content()
            with open("error_page_content.html", "w") as f:
                f.write(content)
            raise

    async def click_frame_close(self):
        await self._visited[-1].locator("iframe[id=\"OverlayIFrame\"]").content_frame.get_by_label("Close image").click()

class BingService(GoogleService):

    def __init__(self, page: Page):
        super().__init__(page, "bing.com")

    async def search_query(self, query: str):
        await self.page.wait_for_selector("input[name='q']", timeout=30000)
        await self.page.fill("input[name='q']", query)
        await self.page.press("input[name='q']", "Enter")


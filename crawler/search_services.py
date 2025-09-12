import logging
from abc import ABC, abstractmethod
from enum import Enum

import playwright
from playwright.async_api import Page


class SearchServices(str, Enum):
    """
    Enum representing supported search service domains.
    """
    BING = "bing.com"


def get_service_by_host(page: Page, host: str):
    """
    Factory method that returns the appropriate SearchService
    subclass instance based on the host domain.

    Args:
        page (Page): Playwright page object.
        host (str): Host domain string, e.g. 'bing.com'.

    Returns:
        SearchService: Instance of a SearchService subclass.
    """
    domain = host.split('.')[0]
    match domain:
        case 'bing':
            return BingService(page)
        case 'google':
            return GoogleService(page, host)
        case _:
            return GoogleService(page, "google.com")


class SearchService(ABC):
    """
    Abstract base class defining common interface and properties for search services.

    Concrete subclasses must implement core asynchronous methods for
    search query, filter interaction, and frame closing.
    """

    _host: str
    _page: Page
    _schema: str
    _test_id_attr: str
    _visited: list[Page]

    def __init__(self, page: Page, host: str, schema: str = "https", test_id_attr: str = ""):
        """
        Initialize SearchService with browser page, host URL components,
        and optional test id attribute for selectors.
        """
        self._host = host
        self._page = page
        self._schema = schema
        self._test_id_attr = test_id_attr
        self._visited = []

    @property
    def host(self):
        """Returns the service host domain."""
        return self._host

    @property
    def page(self):
        """Returns Playwright page object."""
        return self._page

    @property
    def schema(self):
        """Returns the URL schema (typically 'https')."""
        return self._schema

    @property
    def test_id_attr(self):
        """Returns the attribute used for element test IDs."""
        return self._test_id_attr

    @property
    def visited(self):
        """Returns list of pages/tabs visited during crawling."""
        return self._visited

    async def navigate_host(self):
        """
        Navigate Playwright page to the base search URL (schema + host).
        """
        await self.page.goto(self.search_url(), wait_until="networkidle")

    @abstractmethod
    async def search_query(self, query):
        """
        Abstract method to run a search for given query string
        on the underlying search service.
        """
        raise NotImplementedError("'search_query' is not implemented !")

    @abstractmethod
    async def click_obj_filter(self, search_obj_type: str) -> Page:
        """
        Abstract method to select and click on a results filter,
        e.g. clicking 'Images' tab to filter results by images.

        Returns:
            Page: New Playwright page/tab after clicking filter.
        """
        raise NotImplementedError("'click_obj_filter' is not implemented !")

    @abstractmethod
    async def click_frame_close(self):
        """
        Abstract method to close any modal or overlay frame,
        for example image preview frames in search results.
        """
        raise NotImplementedError("'click_frame_close' is not implemented !")

    def search_url(self):
        """
        Construct base URL string for search services.

        Returns:
            str: Full URL string including schema and host.
        """
        return f"{self.schema}://{self.host}"


class GoogleService(SearchService):
    """
    Concrete implementation of SearchService for Google.

    Implements navigation, search query, filter clicking, and frame closing.
    """

    def __init__(self, page: Page, host: str):
        """
        Initialize GoogleService with page, host, and test id attribute.
        """
        super().__init__(page, host, schema="https", test_id_attr='data-bm')

    async def search_query(self, query: str):
        """
        Fill the search input box, submit query, and wait for results.

        Args:
            query (str): Search keyword string.
        """
        await self.page.wait_for_selector("textarea[name='q']", timeout=30000)
        await self.page.fill("textarea[name='q']", query)
        await self.page.press("textarea[name='q']", "Enter")

    async def click_obj_filter(self, search_obj_type: str):
        """
        Click the filter for a specific search object type, e.g. 'Images'.

        Opens result in new page/tab, waits for loads, and returns page object.

        Args:
            search_obj_type (str): Object type filter name (case insensitive).

        Returns:
            Page: New Playwright page after clicking filter.
        """
        page1 = None
        try:
            async with self.page.expect_popup() as page1_info:
                await self.page.locator(
                    f'//nav[@class="b_scopebar"]/ul/li[@id="b-scopeListItem-{search_obj_type.lower()}"]/a').click(timeout=1000)
            page1 = await page1_info.value
            await page1.wait_for_selector('#b_content')

        except Exception as e:
            logging.error(f"Error clicking {search_obj_type} filter link: {e}")
            await self.page.screenshot(path=f"error_{search_obj_type}_filter_click.png")
            content = await self.page.content()
            with open("error_page_content.html", "w") as f:
                f.write(content)
        finally:
            if page1 is None:
                page1 = self.page
            # raise
            self._visited.append(page1)
            return page1

    async def click_frame_close(self, timeout=1000):
        """
        Click the close button on overlay frames such as image previews.

        Args:
            timeout (int): Timeout in ms for locating and clicking.

            Logs timeout errors but does not raise exceptions.
        """
        try:
            await self._visited[-1].locator("iframe[id=\"OverlayIFrame\"]").content_frame.get_by_label("Close image").click(timeout=timeout)
        except playwright.async_api.TimeoutError as te:
            logging.error(f"Timeout error on close frame: {te}")

class BingService(GoogleService):
    """
    Concrete implementation of SearchService for Bing.

    Extends GoogleService, overriding search-specific selectors and behavior.
    """

    def __init__(self, page: Page):
        """
        Initialize BingService with page and preset host.
        """
        super().__init__(page, "bing.com")

    async def search_query(self, query: str, timeout: int=60000):
        """
        Fill search input, submit the query for Bing, and wait for results.

        Args:
            query (str): Search keyword string.
            timeout (int): Timeout to wait for selector.
        """
        await self.page.wait_for_selector("input[name='q']", timeout=timeout)
        await self.page.fill("input[name='q']", query)
        await self.page.press("input[name='q']", "Enter")
        # await self.page.reload()


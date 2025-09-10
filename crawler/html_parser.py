import time
from html.parser import HTMLParser


class ImgHTMLParser(HTMLParser):
    """
    A subclass of HTMLParser dedicated to extracting image ('img') tags and their attributes.

    Attributes:
        srcs (dict): A dictionary mapping image 'src' URLs to a metadata dictionary of their attributes.

    Usage:
        Instantiate this parser and feed it HTML content via 'feed()'.
        The 'srcs' dictionary will be populated with image URLs and associated attributes.
    """

    def __init__(self):
        """
        Initialize the parser and the dictionary for source URLs and their metadata.
        """
        super().__init__()
        self.srcs = {}

    def handle_starttag(self, tag, attrs):
        """
        Handle HTML start tags.

        Specifically, detects 'img' tags, extracts their 'src' attribute if it contains 'https',
        and stores all other attributes in a metadata dictionary along with a timestamp.

        Args:
            tag (str): Name of the HTML start tag encountered.
            attrs (list): List of (attribute, value) tuples for this tag.
        """
        # global srcs
        if tag == 'img':
            src = ""
            meta = {"ts": str(time.time())}
            for attr, value in attrs:
                if attr == 'src' and 'https' in value:
                    src = value
                else:
                    meta[attr] = value
            if src:
                self.srcs[src] = meta

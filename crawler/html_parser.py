from html.parser import HTMLParser


class ImgHTMLParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.srcs = {}

    def handle_starttag(self, tag, attrs):
        # global srcs
        if tag == 'img':
            src = ""
            meta = {}
            for attr, value in attrs:
                if attr == 'src' and 'https' in value:
                    src = value
                else:
                    meta[attr] = value
            if src:
                self.srcs[src] = meta

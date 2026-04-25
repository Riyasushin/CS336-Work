from __future__ import annotations

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import bytes_to_str, detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    html = bytes_to_str(html_bytes, encoding)
    return extract_plain_text(html)

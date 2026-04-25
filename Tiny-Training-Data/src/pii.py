from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Covers 10-digit US formats in the test: bare, dashed, spaced, and paren-wrapped.
_PHONE_RE = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def mask_emails(text: str) -> tuple[str, int]:
    return _EMAIL_RE.subn("|||EMAIL_ADDRESS|||", text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return _PHONE_RE.subn("|||PHONE_NUMBER|||", text)


def mask_ips(text: str) -> tuple[str, int]:
    return _IP_RE.subn("|||IP_ADDRESS|||", text)

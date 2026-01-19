from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.parse import urlunparse

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class WebPage:
    url: str
    title: str | None
    text: str


def _is_blocked_host(host: str) -> bool:
    h = (host or "").strip().lower()
    if not h:
        return True

    if h in {"localhost", "0.0.0.0", "127.0.0.1", "::1"}:
        return True

    # If it's a literal IP, block private/loopback/link-local/etc.
    try:
        ip = ipaddress.ip_address(h)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return True
    except ValueError:
        pass

    # Common internal-only hostname suffixes.
    if h.endswith(".local") or h.endswith(".internal"):
        return True

    return False


def is_safe_public_url(url: str) -> bool:
    try:
        u = urlparse(url)
    except Exception:
        return False

    if u.scheme not in {"http", "https"}:
        return False

    host = u.hostname or ""
    if _is_blocked_host(host):
        return False

    return True


def _maybe_rewrite_reddit_url(url: str) -> str:
    """Rewrite Reddit URLs to old.reddit.com for server-rendered HTML.

    Modern Reddit pages are often heavily client-rendered, which results in near-empty
    HTML for scraping. old.reddit.com provides usable HTML for extracting post/comment text.
    """

    try:
        u = urlparse(url)
    except Exception:
        return url

    host = (u.hostname or "").lower()
    if not host:
        return url

    if host in {"reddit.com", "www.reddit.com", "m.reddit.com"}:
        # Preserve path/query/fragment.
        return urlunparse((u.scheme or "https", "old.reddit.com", u.path or "", u.params or "", u.query or "", u.fragment or ""))

    return url


def _extract_reddit_text(html: str) -> tuple[str | None, str]:
    """Extract readable post + comment text from old Reddit HTML."""

    if not html:
        return (None, "")

    soup = BeautifulSoup(html, "html.parser")

    # Drop non-content sections.
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()

    title = None
    a = soup.select_one("a.title")
    if a and a.get_text(strip=True):
        title = a.get_text(strip=True)[:200] or None
    elif soup.title and soup.title.string:
        title = str(soup.title.string).strip()[:200] or None

    parts: list[str] = []

    post_md = soup.select_one("div.expando div.usertext-body div.md")
    if post_md:
        post_text = post_md.get_text("\n", strip=True)
        post_text = re.sub(r"\n{3,}", "\n\n", (post_text or "").strip())
        if post_text:
            parts.append("Post:\n" + post_text)

    # Grab a handful of comment bodies (old Reddit keeps them in .comment .md).
    comment_mds = soup.select("div.comment div.entry div.usertext-body div.md")
    comments: list[str] = []
    for md in comment_mds[:20]:
        txt = md.get_text("\n", strip=True)
        txt = re.sub(r"\n{3,}", "\n\n", (txt or "").strip())
        # Skip ultra-short noise.
        if not txt or len(txt) < 40:
            continue
        comments.append(txt)
        if len(comments) >= 8:
            break

    if comments:
        joined = "\n\n".join(f"- {c}" for c in comments)
        parts.append("Top comments:\n" + joined)

    text = "\n\n".join(parts).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    # If we couldn't find anything meaningful, return empty so callers can fall back.
    if len(text) < 160:
        return (title, "")
    return (title, text)


async def fetch_html(
    url: str,
    *,
    timeout_s: float = 15.0,
    max_bytes: int = 800_000,
) -> tuple[str | None, str | None]:
    """Fetch HTML content with a hard size cap.

    Returns (final_url, html_text) or (None, None) on failure.
    """

    if not is_safe_public_url(url):
        return (None, None)

    timeout = httpx.Timeout(timeout_s)
    headers = {
        # Use a browser-like UA; some sites (notably Reddit) block overly-generic agents.
        "User-Agent": "Mozilla/5.0 (compatible; OSRS-AI-Wiki-Chat/0.1; +https://oldschool.runescape.wiki)",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            async with client.stream("GET", url) as r:
                r.raise_for_status()

                ctype = (r.headers.get("Content-Type") or "").lower()
                if "text/html" not in ctype and "application/xhtml" not in ctype and "text/plain" not in ctype:
                    return (str(r.url), None)

                # Respect Content-Length if present.
                try:
                    cl = int(r.headers.get("Content-Length") or "0")
                    if cl and cl > int(max_bytes):
                        return (str(r.url), None)
                except Exception:
                    pass

                buf = bytearray()
                async for chunk in r.aiter_bytes():
                    if not chunk:
                        continue
                    remaining = int(max_bytes) - len(buf)
                    if remaining <= 0:
                        break
                    buf.extend(chunk[:remaining])

        if not buf:
            return (None, None)

        # httpx will guess encoding; but we have bytes. Decode defensively.
        text = buf.decode("utf-8", errors="ignore")
        return (str(r.url), text)
    except Exception:
        return (None, None)


def extract_readable_text(html: str) -> tuple[str | None, str]:
    """Extract readable text from HTML.

    This is intentionally lightweight (no heavy readability dependency).
    """

    if not html:
        return (None, "")

    soup = BeautifulSoup(html, "html.parser")

    title = None
    if soup.title and soup.title.string:
        title = str(soup.title.string).strip()[:200] or None

    # Drop non-content sections.
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()

    # Prefer main/article if present.
    main = soup.find("main") or soup.find("article") or soup.body or soup

    # Remove common boilerplate containers.
    for sel in ["nav", "header", "footer", "aside"]:
        for tag in main.find_all(sel):
            tag.decompose()

    text = main.get_text("\n", strip=True)

    # Normalize whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    # Drop ultra-short pages.
    if len(text) < 200:
        return (title, "")

    return (title, text)


def chunk_text(text: str, *, max_chars: int = 900, overlap: int = 120) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not text:
        return []

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        parts.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return parts


async def fetch_and_extract_page(
    url: str,
    *,
    timeout_s: float = 15.0,
    max_bytes: int = 800_000,
) -> WebPage | None:
    url = _maybe_rewrite_reddit_url(url)

    final_url, html = await fetch_html(url, timeout_s=timeout_s, max_bytes=max_bytes)
    if not final_url or not html:
        return None

    host = (urlparse(final_url).hostname or "").lower()
    if host.endswith("reddit.com"):
        title, text = _extract_reddit_text(html)
        if not text:
            # Fall back to generic extraction if reddit-specific parsing failed.
            title, text = extract_readable_text(html)
    else:
        title, text = extract_readable_text(html)
    if not text:
        return None

    return WebPage(url=final_url, title=title, text=text)

#!/usr/bin/env python3
"""
Scrape Floor & Decor product metadata & images for the
San Leandro store (storeID=238).

Outputs:
  - data/san_leandro_products.csv
  - images/<SKU>.jpg

Run from repo root:
  .\.venv\Scripts\Activate.ps1
  python scrape_238_products.py
"""

import csv
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------- CONFIG ----------------------

BASE_URL = "https://www.flooranddecor.com"
STORE_ID = 238  # San Leandro

# Top-level categories to scan
CATEGORY_SLUGS = [
    "/tile",
    "/wood",
    "/vinyl",
    "/laminate",
    "/stone",
    "/decoratives",
    "/fixtures",
    "/installation-materials",
]

# Product URL pattern:
# e.g. https://www.flooranddecor.com/...-101363893.html
PRODUCT_URL_RE = re.compile(
    r"https://www\.flooranddecor\.com/[A-Za-z0-9_\-/]+-(\d{6,})\.html"
)

DATA_DIR = "data"
IMAGES_DIR = "images"
METADATA_CSV = os.path.join(DATA_DIR, "san_leandro_products.csv")

REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 0.5  # seconds between requests

# ---------------------- LOGGING ----------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------- HTTP HELPERS ----------------------


def make_session() -> requests.Session:
    """Create a requests session with a decent User-Agent."""
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


def set_store_context(session: requests.Session) -> None:
    """
    Hit the store selector URL so cookies / context are set
    for the San Leandro store (ID=238).
    """
    url = f"{BASE_URL}/store?storeID={STORE_ID}"
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        logging.info("Store context response: %s %s", r.status_code, url)
    except Exception as e:
        logging.warning("Failed to set store context: %s", e)

# ---------------------- URL DISCOVERY ----------------------


def fetch_product_urls_for_category(
    session: requests.Session, category_slug: str
) -> Set[str]:
    """
    Given a category slug like "/tile", fetch the main category page
    and extract product detail URLs using PRODUCT_URL_RE.
    """
    url = urljoin(BASE_URL, category_slug)
    logging.info("Scanning category page: %s", url)

    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        logging.warning("Failed to fetch category %s: %s", url, e)
        return set()

    if r.status_code != 200:
        logging.warning("Category %s returned status %s", url, r.status_code)
        return set()

    html = r.text

    found_urls: Set[str] = set()
    for match in PRODUCT_URL_RE.finditer(html):
        full_url = match.group(0)
        found_urls.add(full_url)

    logging.info(
        "Category %s -> %d candidate product URLs",
        category_slug,
        len(found_urls),
    )
    return found_urls


def fetch_all_product_urls(session: requests.Session) -> Dict[str, Set[str]]:
    """
    For each category, collect product URLs.
    Returns {category_slug: {url1, url2, ...}}.
    """
    result: Dict[str, Set[str]] = {}
    for slug in CATEGORY_SLUGS:
        urls = fetch_product_urls_for_category(session, slug)
        if urls:
            result[slug] = urls
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    total = sum(len(v) for v in result.values())
    logging.info("Total unique product URLs found across categories: %d", total)
    return result

# ---------------------- PARSING HELPERS ----------------------


def extract_sku_from_text(text: str) -> Optional[str]:
    """Look for 'SKU: 101363893' style patterns in the raw HTML."""
    m = re.search(r"SKU[:\s]+(\d{6,})", text)
    if m:
        return m.group(1)
    return None


def extract_sku_from_url(url: str) -> Optional[str]:
    """Fallback: get SKU from the trailing '-digits.html' in the URL."""
    m = re.search(r"-(\d{6,})\.html", url)
    if m:
        return m.group(1)
    return None


def extract_product_image_url(soup: BeautifulSoup, sku: Optional[str]) -> Optional[str]:
    """
    Try several strategies to get the *actual* product hero image.

    Order:
      1) Any <img> whose src contains the SKU (most specific)
      2) JSON-LD Product schema "image" field
      3) <meta property="og:image"> (but skip generic nav banners)
      4) Heuristic scan of remaining <img> tags (skipping logos/icons/banners)
    """

    def is_bad_image_url(url: str) -> bool:
        """Filter out obvious non-product or generic banner assets."""
        low = url.lower()
        # Generic/global stuff we never want
        bad_tokens = [
            "logo",
            "icon",
            "sprite",
            "facebook",
            "twitter",
            "pinterest",
            "instagram",
            "youtube",
            "favicon",
        ]
        if any(t in low for t in bad_tokens):
            return True

        # The generic bathroom inspiration banner that was showing up everywhere
        if "bathroom_inspriationnavigation" in low or "inspirationnavigation" in low:
            return True

        return False

    # 1) If we know the SKU, look for img src that contains it
    if sku:
        for img in soup.find_all("img"):
            src = (
                img.get("src")
                or img.get("data-src")
                or img.get("data-lazy")
                or img.get("data-original")
            )
            if not src:
                continue
            src = src.strip()
            if not src or src.startswith("data:"):
                continue

            full = urljoin(BASE_URL, src)
            if sku in full and not is_bad_image_url(full):
                return full

    # 2) JSON-LD Product schema with an "image" field
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = script.string or script.get_text(strip=True)
            if not raw:
                continue
            data = json.loads(raw)
        except Exception:
            continue

        def find_product_image(obj):
            if isinstance(obj, dict):
                if obj.get("@type") == "Product" and obj.get("image"):
                    img = obj["image"]
                    if isinstance(img, list):
                        img = img[0]
                    return img
                for v in obj.values():
                    res = find_product_image(v)
                    if res:
                        return res
            elif isinstance(obj, list):
                for v in obj:
                    res = find_product_image(v)
                    if res:
                        return res
            return None

        img = find_product_image(data)
        if img:
            full = urljoin(BASE_URL, img)
            if not is_bad_image_url(full):
                return full

    # 3) og:image – skip if it's a generic banner
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        candidate = og["content"].strip()
        if candidate:
            full = urljoin(BASE_URL, candidate)
            if not is_bad_image_url(full):
                return full

    # 4) Fallback: scan <img> tags and pick a reasonable candidate
    candidates: List[str] = []
    for img in soup.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy")
            or img.get("data-original")
        )
        if not src:
            continue
        src = src.strip()
        if not src or src.startswith("data:"):
            continue

        full = urljoin(BASE_URL, src)
        if is_bad_image_url(full):
            continue

        candidates.append(full)

    if candidates:
        return candidates[0]

    return None

# ---------------------- PRODUCT PAGE PARSING ----------------------


def parse_product_page(
    session: requests.Session, url: str, category_slug: str
) -> Optional[Dict[str, str]]:
    """Fetch and parse a single product page into a metadata dict."""
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        logging.warning("Failed to fetch product %s: %s", url, e)
        return None

    if r.status_code != 200:
        logging.warning("Product %s returned status %s", url, r.status_code)
        return None

    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    # --- SKU ---
    sku = extract_sku_from_text(html)
    if not sku:
        sku = extract_sku_from_url(url)
    if not sku:
        logging.debug("Could not extract SKU from %s", url)
        return None

    # --- Name ---
    name: Optional[str] = None
    h1 = soup.find("h1")
    if h1:
        name = h1.get_text(strip=True)

    if not name:
        meta_title = soup.find("meta", property="og:title")
        if meta_title and meta_title.get("content"):
            name = meta_title["content"].strip()

    if not name:
        # Fallback: use URL slug
        name = (
            url.split("/")[-1]
            .split(".html")[0]
            .replace("-", " ")
            .title()
        )

    # --- Image URL ---
    image_url = extract_product_image_url(soup, sku)
    logging.debug("SKU %s -> image_url: %s", sku, image_url)

    return {
        "sku": sku,
        "name": name,
        "category_slug": category_slug,
        "product_url": url,
        "image_url": image_url or "",
        # image_filename will be filled in later by download_images()
    }


def scrape_products(
    session: requests.Session, product_urls_by_category: Dict[str, Set[str]]
) -> List[Dict[str, str]]:
    """Scrape all product pages and return a list of metadata dicts."""
    rows: List[Dict[str, str]] = []
    seen_skus: Set[str] = set()

    for category, urls in product_urls_by_category.items():
        logging.info("Scraping %d products for category %s", len(urls), category)
        for url in urls:
            meta = parse_product_page(session, url, category)
            if not meta:
                continue

            sku = meta["sku"]
            if sku in seen_skus:
                continue
            seen_skus.add(sku)

            rows.append(meta)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    logging.info("Parsed %d unique products successfully", len(rows))
    return rows

# ---------------------- OUTPUT HELPERS ----------------------


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def save_metadata(rows: List[Dict[str, str]], csv_path: str) -> None:
    if not rows:
        logging.info("No rows to save, skipping CSV write.")
        return

    fieldnames = [
        "sku",
        "name",
        "category_slug",
        "product_url",
        "image_url",
        "image_filename",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = row.copy()
            out_row.setdefault("image_filename", "")
            writer.writerow(out_row)

    logging.info("Metadata written to %s (%d rows)", csv_path, len(rows))


def download_images(session: requests.Session, rows: List[Dict[str, str]]) -> None:
    if not rows:
        logging.info("No rows for image download.")
        return

    logging.info("Downloading images for %d products...", len(rows))
    for row in rows:
        sku = row["sku"]
        img_url = row.get("image_url")
        if not img_url:
            logging.info("Skipping SKU %s (no image_url)", sku)
            row["image_filename"] = ""
            continue

        img_path = os.path.join(IMAGES_DIR, f"{sku}.jpg")
        row["image_filename"] = os.path.basename(img_path)

        if os.path.exists(img_path):
            logging.info("Image already exists for SKU %s, skipping", sku)
            continue

        logging.info("Downloading image for SKU %s: %s", sku, img_url)
        try:
            resp = session.get(img_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and resp.content:
                with open(img_path, "wb") as f:
                    f.write(resp.content)
            else:
                logging.warning(
                    "Image download failed for SKU %s (%s): status %s",
                    sku,
                    img_url,
                    resp.status_code,
                )
        except Exception as e:
            logging.warning(
                "Error downloading image for SKU %s (%s): %s",
                sku,
                img_url,
                e,
            )

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    logging.info("Image download step complete.")

# ---------------------- MAIN ----------------------


def main() -> None:
    ensure_dirs()
    session = make_session()

    # Scope to San Leandro store
    set_store_context(session)

    logging.info("Discovering product URLs from category pages...")
    product_urls_by_category = fetch_all_product_urls(session)

    logging.info("Starting product scrape...")
    products = scrape_products(session, product_urls_by_category)

    logging.info("Downloading images...")
    download_images(session, products)

    logging.info("Saving metadata...")
    save_metadata(products, METADATA_CSV)

    logging.info("Done. Metadata -> %s, images -> %s/", METADATA_CSV, IMAGES_DIR)


if __name__ == "__main__":
    main()

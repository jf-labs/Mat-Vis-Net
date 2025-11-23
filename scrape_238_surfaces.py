#!/usr/bin/env python3
"""
Scrape Floor & Decor *surface* products & images for the
San Leandro store (storeID=238).

We only care about:
  - Tile (porcelain/ceramic/wall)
  - Wood (hardwood / engineered / wood-look)
  - Decorative / deco / ledger / mosaic (via /decoratives)

Outputs:
  - data/san_leandro_products.csv
  - images/<SKU>.jpg

Run from repo root:
  .\.venv\Scripts\Activate.ps1
  python scrape_238_surfaces.py
"""

import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------- CONFIG ----------------------

BASE_URL = "https://www.flooranddecor.com"
STORE_ID = 238  # San Leandro

# Top-level categories to scan – these are the ONLY roots we ever follow.
CATEGORY_SLUGS = [
    "/tile",         # porcelain/ceramic/wall tile
    "/wood",         # hardwood / engineered / wood-look
    "/decoratives",  # mosaics, decorative/deco/ledger, etc.
    "/stone",       # natural stone surfaces
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

SESSION_TIMEOUT = 15
RETRY_DELAY = 2
MAX_RETRIES = 3

DATA_DIR = Path("data")
IMAGES_DIR = Path("images")
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = DATA_DIR / "san_leandro.csv"

# Product URL regex – captures SKU from URLs like
# /tile/whatever-name-101234567.html
PRODUCT_URL_RE = re.compile(
    r"^/(?P<cat>[^/]+)/(?P<name_slug>.+)-(?P<sku>[A-Z0-9]{5,}).html$"
)

# These are words that strongly suggest a tile/wood/deco *surface*,
# NOT cabinets, grout, mortar, tools, etc.
SURFACE_KEYWORDS = {
    "tile", "porcelain", "ceramic", "marble", "travertine", "granite",
    "quartzite", "slate", "limestone", "mosaic", "ledger", "stacked stone",
    "paver", "plank", "hardwood", "engineered hardwood", "laminate",
    "vinyl plank", "luxury vinyl plank", "lvp", "lvt", "wood-look",
}

# Words that scream "this is NOT a surface" – skip aggressively.
NON_SURFACE_KEYWORDS = {
    "cabinet", "drawer", "valance", "shelf", "filler", "toe kick",
    "pull", "knob", "handle", "hardware", "hinge", "organizer",
    "trash", "cutlery", "kit", "tape", "painted",
    "mortar", "thinset", "grout", "adhesive", "spacer", "trowel",
    "saw", "blade", "spackle", "patch", "primer", "roller",
    "membrane", "underlayment", "backer", "leveler", "cement",
    "cleaner", "sealant", "sealer", "polish",
}


# ---------------------- UTILS ----------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    s.timeout = SESSION_TIMEOUT
    return s


def safe_request(
    session: requests.Session,
    url: str,
    method: str = "GET",
    **kwargs,
) -> Optional[requests.Response]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.request(method, url, timeout=SESSION_TIMEOUT, **kwargs)
            if resp.status_code == 200:
                return resp
            logging.warning("Non-200 (%s) for %s", resp.status_code, url)
        except requests.RequestException as e:
            logging.warning("Request error (%d/%d) %s -> %s", attempt, MAX_RETRIES, url, e)
        time.sleep(RETRY_DELAY)
    logging.error("Giving up on %s after %d attempts", url, MAX_RETRIES)
    return None


def safe_get(session: requests.Session, url: str) -> Optional[BeautifulSoup]:
    resp = safe_request(session, url, "GET")
    if resp is None:
        return None
    return BeautifulSoup(resp.text, "html.parser")


# ---------------------- PRODUCT FILTERING ----------------------


def looks_like_surface_product(name: str, breadcrumbs: List[str]) -> bool:
    """
    Decide if this product is actually a surface (tile/wood/deco) vs. cabinets,
    hardware, tools, etc., based on name + breadcrumb text.
    """
    text = " ".join([name] + breadcrumbs).lower()

    # If any explicit "non-surface" keyword appears, toss it.
    if any(bad in text for bad in NON_SURFACE_KEYWORDS):
        return False

    # Require at least one surface keyword.
    if any(good in text for good in SURFACE_KEYWORDS):
        return True

    # Weak fallback: if the breadcrumb chain contains "tile" or "wood", keep.
    if "tile" in text or "wood" in text or "mosaic" in text or "ledger" in text:
        return True

    return False


def parse_product_json_blob(soup: BeautifulSoup) -> Dict:
    """
    Floor & Decor has a dataLayer / window.__FND__ style JSON blob that
    contains structured product info. We try to grab it if present.
    """
    # Look for a <script> tag that contains "productData" or "productDetails" JSON.
    for script in soup.find_all("script"):
        if not script.string:
            continue
        text = script.string.strip()
        if "productData" in text or "productDetails" in text:
            # Very loose JSON extractor – we just grab the first {...} block.
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                continue
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return {}


def parse_product_page(session: requests.Session, product_url: str) -> Optional[Dict]:
    soup = safe_get(session, product_url)
    if soup is None:
        return None

    # Basic fields
    title_tag = soup.find("h1")
    name = title_tag.get_text(strip=True) if title_tag else ""

    # Breadcrumb trail for extra context
    breadcrumbs = []
    bc = soup.find("ul", class_="breadcrumbs")
    if bc:
        for li in bc.find_all("li"):
            txt = li.get_text(" ", strip=True)
            if txt:
                breadcrumbs.append(txt)

    # Hard filter by name + breadcrumbs
    if not looks_like_surface_product(name, breadcrumbs):
        return None

    # Extract SKU from URL
    parsed = urlparse(product_url)
    m = PRODUCT_URL_RE.match(parsed.path)
    if not m:
        return None
    sku = m.group("sku")

    # Try to find price and some attributes
    price = None
    price_el = soup.find(attrs={"data-product-price": True})
    if price_el:
        try:
            price = float(price_el["data-product-price"])
        except (KeyError, ValueError, TypeError):
            pass

    # Try to parse the product JSON blob for category, material, etc.
    meta = parse_product_json_blob(soup)
    material = None
    surface_type = None
    uom = None

    if meta:
        # This is intentionally loose; you can adjust keys after inspecting real blobs.
        material = (
            meta.get("material")
            or meta.get("Material")
            or meta.get("attributes", {}).get("Material")
        )
        surface_type = (
            meta.get("productType")
            or meta.get("productSubType")
            or meta.get("attributes", {}).get("Product Subtype")
        )
        uom = (
            meta.get("uom")
            or meta.get("unitOfMeasure")
            or meta.get("attributes", {}).get("Unit of Measure")
        )

    # Grab main image URL
    img_url = None
    img_tag = soup.find("img", id="primary-image") or soup.find("img", class_="primary-image")
    if img_tag and img_tag.get("src"):
        img_url = img_tag["src"]
    else:
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            img_url = og["content"]

    return {
        "sku": sku,
        "name": name,
        "breadcrumbs": " > ".join(breadcrumbs),
        "url": product_url,
        "price": price,
        "material": material,
        "surface_type": surface_type,
        "uom": uom,
        "image_url": img_url,
    }


# ---------------------- CATEGORY / LISTING CRAWL (HARD LOCKED TO ROOT) ----------------------


def fetch_product_urls_for_category(
    session: requests.Session,
    category_url: str,
    max_pages: int = 10000,
) -> List[str]:
    """
    Walk the pagination for a single *surface* category URL and collect product URLs.

    IMPORTANT:
    - We ONLY follow links that stay under the same top-level category path
      as `category_url` (e.g. "/tile", "/wood", "/decoratives").
    - This prevents us from wandering into other departments like
      semi-custom cabinets, fixtures, etc.
    """
    visited_pages: Set[str] = set()
    product_urls: Set[str] = set()

    # Normalize the starting URL and compute its path prefix
    start_url = urljoin(BASE_URL, category_url)
    root_path = urlparse(start_url).path  # e.g. "/tile"

    # Small FIFO queue for BFS over ONLY this category's pages
    to_visit: List[str] = [start_url]
    pages_seen = 0

    while to_visit and pages_seen < max_pages:
        current_url = to_visit.pop(0)

        # Paranoia guard: if for any reason we leave the root path, drop it.
        current_path = urlparse(current_url).path
        if not current_path.startswith(root_path):
            continue

        if current_url in visited_pages:
            continue
        visited_pages.add(current_url)

        logging.info("Scanning category page (%d pages seen) %s", pages_seen, current_url)
        pages_seen += 1

        soup = safe_get(session, current_url)
        if soup is None:
            continue

        # 1) Grab product tiles on this page
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("/"):
                continue

            # Product URLs look like:
            #   /<category>/<product-name>-<SKU>.html
            #   /<category>/<product-name>-<SKU>.html?dwvar_<SKU>_color=...
            m = PRODUCT_URL_RE.match(urlparse(href).path)
            if not m:
                continue

            sku = m.group("sku")
            product_path = urlparse(href).path

            # Only keep products that ALSO live under the same root_path.
            # This prevents tile pages from pulling in cabinets from some
            # completely different department.
            if not product_path.startswith(root_path):
                continue

            full_product_url = urljoin(BASE_URL, href)
            product_urls.add(full_product_url)

        # 2) Discover *more pages within this same category* (pagination, filters, etc.)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("/"):
                continue

            next_url = urljoin(BASE_URL, href)
            parsed = urlparse(next_url)

            # We only care about links that:
            # - Stay under the same root_path ("/tile", "/wood", "/decoratives")
            if not parsed.path.startswith(root_path):
                continue

            # Skip obvious product detail pages
            if PRODUCT_URL_RE.match(parsed.path):
                continue

            # This is still a "category-like" page: pagination or filtered view
            if next_url not in visited_pages and next_url not in to_visit:
                to_visit.append(next_url)

    return sorted(product_urls)


def fetch_all_product_urls(session: requests.Session) -> List[str]:
    all_urls: Set[str] = set()
    for slug in CATEGORY_SLUGS:
        logging.info("=== Crawling category root: %s ===", slug)
        urls = fetch_product_urls_for_category(session, slug)
        logging.info("Found %d product URLs under %s", len(urls), slug)
        all_urls.update(urls)
    logging.info("Total unique product URLs across all surface categories: %d", len(all_urls))
    return sorted(all_urls)


# ---------------------- IMAGE DOWNLOAD ----------------------


def download_image(session: requests.Session, sku: str, img_url: str) -> Optional[str]:
    """
    Download the main product image to images/<SKU>.jpg if not already present.
    Returns the relative filename or None.
    """
    if not img_url:
        return None

    ext = ".jpg"
    parsed = urlparse(img_url)
    basename = os.path.basename(parsed.path)
    if "." in basename:
        ext = "." + basename.split(".")[-1].split("?")[0]

    filename = f"{sku}{ext}"
    dest_path = IMAGES_DIR / filename
    if dest_path.exists():
        return filename

    resp = safe_request(session, img_url, "GET", stream=True)
    if resp is None:
        return None

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return filename


# ---------------------- MAIN SCRAPE ----------------------


def main() -> None:
    setup_logging()
    session = make_session()

    logging.info("Collecting surface product URLs only (tile/wood/decoratives)...")
    product_urls = fetch_all_product_urls(session)
    logging.info("Total surface-like product URLs to inspect: %d", len(product_urls))

    fieldnames = [
        "sku",
        "name",
        "breadcrumbs",
        "url",
        "price",
        "material",
        "surface_type",
        "uom",
        "image_url",
        "image_filename",
    ]

    rows = []

    for idx, url in enumerate(product_urls, start=1):
        logging.info("(%d/%d) Parsing product %s", idx, len(product_urls), url)
        data = parse_product_page(session, url)
        if data is None:
            continue

        # Download image if available
        img_filename = None
        if data.get("image_url"):
            img_filename = download_image(session, data["sku"], data["image_url"])

        data["image_filename"] = img_filename
        rows.append(data)

    logging.info("Writing %d filtered surface products to %s", len(rows), OUTPUT_CSV)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Done.")


if __name__ == "__main__":
    main()

import csv
import logging
import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------- Config ----------

BASE_URL = "https://www.flooranddecor.com"
STORE_ID = 238  # San Leandro

# Top-level categories to start from (home pages only: no pagination for now)
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

# Regex that matches product detail URLs like:
# https://www.flooranddecor.com/nucore-performance-flooring/...-101363893.html
PRODUCT_URL_RE = re.compile(
    r"https://www\.flooranddecor\.com/[A-Za-z0-9_\-/]+-(\d{6,})\.html"
)

DATA_DIR = "data"
IMAGES_DIR = "images"
METADATA_CSV = os.path.join(DATA_DIR, "san_leandro_products.csv")

REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 0.5  # seconds, to be polite


# ---------- Logging ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------- HTTP helpers ----------

def make_session() -> requests.Session:
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
    """Hit the store page once so the site knows we care about store 238."""
    url = f"{BASE_URL}/store?storeID={STORE_ID}"
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    logging.info("Store context response: %s %s", r.status_code, url)


# ---------- URL discovery ----------

def fetch_product_urls_for_category(
    session: requests.Session, category_slug: str
) -> set[str]:
    """
    Given a category slug like "/tile", return a set of product detail URLs
    found on that page (no pagination for now).
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

    found_urls = set()
    for match in PRODUCT_URL_RE.finditer(html):
        full_url = match.group(0)
        found_urls.add(full_url)

    logging.info(
        "Category %s -> %d candidate product URLs",
        category_slug,
        len(found_urls),
    )
    return found_urls


def fetch_all_product_urls(session: requests.Session) -> dict[str, set[str]]:
    """
    For each category, collect product URLs.
    Returns a dict: {category_slug: {url1, url2, ...}}
    """
    result: dict[str, set[str]] = {}
    for slug in CATEGORY_SLUGS:
        urls = fetch_product_urls_for_category(session, slug)
        if urls:
            result[slug] = urls
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    total = sum(len(v) for v in result.values())
    logging.info("Total unique product URLs found across categories: %d", total)
    return result


# ---------- Product page parsing ----------

def extract_sku_from_text(text: str) -> str | None:
    m = re.search(r"SKU[:\s]+(\d{6,})", text)
    if m:
        return m.group(1)
    return None


def extract_sku_from_url(url: str) -> str | None:
    m = re.search(r"-(\d{6,})\.html", url)
    if m:
        return m.group(1)
    return None


def parse_product_page(
    session: requests.Session, url: str, category_slug: str
) -> dict | None:
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

    # SKU: try HTML text then fallback to URL
    sku = extract_sku_from_text(html)
    if not sku:
        sku = extract_sku_from_url(url)
    if not sku:
        logging.debug("Could not extract SKU from %s", url)
        return None

    # Product name
    name = None
    h1 = soup.find("h1")
    if h1:
        name = h1.get_text(strip=True)

    if not name:
        meta_title = soup.find("meta", property="og:title")
        if meta_title and meta_title.get("content"):
            name = meta_title["content"].strip()

    if not name:
        # Last fallback: use the URL slug
        name = url.split("/")[-1].split(".html")[0].replace("-", " ").title()

    # Main image: use og:image if available
    image_url = None
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        image_url = og_image["content"].strip()

    return {
        "sku": sku,
        "name": name,
        "category_slug": category_slug,
        "product_url": url,
        "image_url": image_url,
    }


def scrape_products(
    session: requests.Session, product_urls_by_category: dict[str, set[str]]
) -> list[dict]:
    rows: list[dict] = []
    seen_skus: set[str] = set()

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


# ---------- Output helpers ----------

def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def save_metadata(rows: list[dict], csv_path: str) -> None:
    if not rows:
        logging.info("No rows to save, skipping CSV write.")
        return

    fieldnames = ["sku", "name", "category_slug", "product_url", "image_url", "image_filename"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logging.info("Metadata written to %s (%d rows)", csv_path, len(rows))


def download_images(session: requests.Session, rows: list[dict]) -> None:
    if not rows:
        logging.info("No rows for image download.")
        return

    logging.info("Downloading images for %d products...", len(rows))
    for row in rows:
        sku = row["sku"]
        img_url = row.get("image_url")
        if not img_url:
            continue

        # Normalize and give each image a simple filename based on SKU
        img_path = os.path.join(IMAGES_DIR, f"{sku}.jpg")
        row["image_filename"] = os.path.basename(img_path)

        if os.path.exists(img_path):
            continue  # already downloaded

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
            logging.warning("Error downloading image for SKU %s (%s): %s", sku, img_url, e)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    logging.info("Image download step complete.")


# ---------- Main ----------

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

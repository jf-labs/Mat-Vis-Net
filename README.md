# Material Visual Network

Image-based product matcher for flooring and tile — given a photo of a plank, tile, or deco piece, return the closest matching SKUs from a catalog.

---

## Getting Started (TL;DR)

### 1. Clone the repo

```bash
git clone <your-repo-url>.git
cd Mat-Vis-Net
```
---

## Overview

Material Visual Network is an experiment in **visual search for building materials**.  
The goal is to take a customer photo (e.g., a phone picture of a plank or tile) and:

- Identify what kind of product it is (LVP, porcelain tile, wood, deco, etc.)
- Suggest visually similar products (color, texture, finish)
- Narrow down to specific SKUs from a store’s catalog

Right now, the project is focused on:

- Building a **clean image + metadata dataset** from a single flooring retailer
- Prototyping a **model that embeds product photos into a visual feature space**
- Using **nearest-neighbor search** to retrieve the closest matching products

---

## Current Status

This repo is **early-stage / WIP** and currently focuses on **data collection**.

Implemented so far:

- Store-specific product scraper  
  - Crawls category pages (e.g., tile, wood, vinyl)
  - Collects product URLs and metadata
  - Downloads primary product images and saves them by SKU
- Basic local file structure for CSVs and images

Planned next steps:

- Clean/dedupe images and handle identical images under different SKUs  
- Build a Jupyter notebook for exploratory analysis and baseline models  
- Train an image embedding model (e.g., CNN / CLIP-style)  
- Implement top-K visual similarity search over the catalog  
- Simple demo: upload an image --> get back likely matching SKUs

---

## Tech Stack (planned & in use)

- **Language:** Python
- **Data Collection:** `requests`, `BeautifulSoup4` (or similar), `pandas`
- **Image Handling:** `Pillow` / `opencv-python` (planned)
- **Modeling (planned):**
  - Pretrained CNN or CLIP-style encoder
  - Metric learning / embedding-based retrieval
  - Nearest neighbor search (e.g., FAISS / sklearn NearestNeighbors)

---

## Repository Structure (WIP)

```text
.
├── scrape_238_products.py     # Script to scrape product metadata + images for a single store
├── data/
│   ├── csv/                   # Raw product metadata (per-category or per-run CSVs)
│   └── images/                # Downloaded product images, typically named <SKU>.jpg
└── README.md

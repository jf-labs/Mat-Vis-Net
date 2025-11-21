# Material Visual Network

Image-based product matcher for flooring and tile — given a photo of a plank, tile, or decorative piece, return the closest matching SKUs from a store catalog (currently Floor & Decor, San Leandro – Store 238).

---

## Getting Started (TL;DR)

### 1. Clone the repository

    git clone <your-repo-url>.git
    cd Mat-Vis-Net

### 2. Set up the Python environment

    python -m venv .venv

Windows:

    .venv\Scripts\activate

macOS / Linux:

    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

### 3. Configure environment variables

If needed, copy the example environment file and adjust values:

    cp .env.example .env

### 4. Scrape products for Store 238

Run the scraper to populate the local dataset:

    python scrape_238_products.py

This will create or update:

- `data/san_leandro_products.csv` – product metadata for Floor & Decor San Leandro (Store 238)
- `images/<SKU>.jpg` – primary product images downloaded by SKU

### 5. Run the prototype notebook (optional)

Launch Jupyter and explore the prototype:

    jupyter notebook notebooks/mvn_prototype.ipynb

### 6. Frontend client (early WIP)

The `apps/client` directory is reserved for a web UI.

    cd apps/client
    npm install
    npm run dev   # or npm run start, depending on package.json

---

## Overview

Material Visual Network is an experiment in visual search for building materials.

The goal is to take a customer photo (for example, a phone picture of a plank or tile) and:

- Identify what kind of product it is (LVP, porcelain tile, wood, decorative, etc.).
- Suggest visually similar products based on color, texture, pattern, and finish.
- Narrow results down to specific SKUs from a store’s catalog.
- Provide a building block for an assistant that can answer questions such as:
  - “What in-stock products look like this?”
  - “What is the closest match to this plank in the San Leandro store?”

The current focus is on:

- Building a clean image + metadata dataset from a single Floor & Decor location.
- Prototyping a model that embeds product photos into a visual feature space.
- Using nearest-neighbor search to retrieve the closest matching products.

---

## Algorithm

This project uses a content-based image retrieval algorithm built on top of a pretrained vision–language model and a k-Nearest Neighbors (kNN) search over product embeddings.

The algorithm operates in three main stages:

1. **Product Image Embedding**

   - Each product image from the catalog is passed through a pretrained CLIP-style vision encoder.
   - The encoder maps the image into a fixed-length vector (an embedding) in a high-dimensional space.
   - These embeddings capture visual semantics such as color, texture, and pattern, rather than relying solely on textual metadata.

2. **Catalog Index Construction**

   - All product embeddings are stacked into a single matrix and stored locally.
   - A kNN index (for example, scikit-learn’s `NearestNeighbors`) is built over this matrix using cosine distance as the similarity metric.
   - This index allows efficient retrieval of products whose embeddings are closest to a query image in cosine space.

3. **Similarity Search and Post-Filtering**

   - When a query image is provided, the system:
     - Computes an embedding for the query image using the same encoder.
     - Performs a kNN search against the catalog index to retrieve the top-K nearest neighbors.
     - Applies metadata and business filters (for example, matching on material type such as tile vs. wood, or restricting to in-stock products for Store 238).
     - Optionally excludes the exact same SKU so that recommendations surface similar alternatives instead of simply repeating the input product.

In effect, the system performs a vector similarity search over the catalog: visually similar SKUs lie near one another in embedding space, and the kNN lookup returns the best candidates consistent with store constraints.

---

## Current Status

This repository is early-stage and currently emphasizes data collection and prototyping.

Implemented so far:

- Store-specific product scraper:
  - Crawls category pages (tile, wood, vinyl, laminate, and related materials).
  - Collects product URLs and metadata.
  - Downloads primary product images and saves them by SKU.
- Local dataset:
  - `data/san_leandro_products.csv` for product metadata.
  - `images/<SKU>.jpg` for product images.
- Prototype Jupyter notebook:
  - `notebooks/mvn_prototype.ipynb` for exploratory analysis and baseline similarity experiments.
- Initial directory layout for:
  - Future backend / API code (`src/`).
  - Infrastructure configuration (`infra/`).
  - Web client (`apps/client/`).

Planned next steps:

- Clean and deduplicate images, handling identical assets under different SKUs.
- Finalize metadata schema (material, category, in-stock flags, price, coverage, etc.).
- Implement a stable embedding pipeline with CLIP or a similar model.
- Build a robust kNN service for top-K visual similarity search over the catalog.
- Expose an HTTP API for “upload image → get matching SKUs.”
- Integrate the API with the `apps/client` UI for an end-to-end demo.

---

## Tech Stack

**In use**

- Language: Python  
- Data collection: `requests`, `beautifulsoup4`, `pandas`  
- Development and exploration: Jupyter Notebook  

**Planned / in progress**

- Image embeddings: CLIP-style pretrained vision encoder  
- Similarity search: scikit-learn `NearestNeighbors` (with potential FAISS integration later)  
- Backend API: FastAPI or similar Python web framework  
- Frontend: React-based client in `apps/client`  
- Infrastructure: Configuration and deployment assets under `infra/`  

---

## Repository Structure

Reflecting the current layout:

```text
.
├── .venv/                       # Local Python virtual environment (not committed)
├── apps/
│   └── client/
│       ├── package.json         # Frontend dependencies and scripts
│       ├── src/
│       │   └── api/             # API client helpers (planned / WIP)
│       └── index.html           # Frontend entry HTML
├── data/
│   ├── san_leandro_products.csv # Scraped product metadata for Store 238
│   └── images/                  # Downloaded product images named by SKU
├── infra/                       # Infrastructure and deployment configuration (WIP)
├── notebooks/
│   └── mvn_prototype.ipynb      # Prototype notebook for embeddings and retrieval
├── src/                         # Backend / API / model code (planned / WIP)
├── venv/                        # Alternative or legacy virtual environment directory
├── .env.example                 # Example environment configuration
├── .gitignore                   # Git ignore rules
├── flow.drawio.png              # System / data flow diagram
├── scrape_238_products.py       # Scraper for Floor & Decor Store 238
└── README.md                    # Project documentation (this file)

# %%
# === Paths and metadata loading ===
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("..").resolve()
CSV_PATH = PROJECT_ROOT / "data" / "san_leandro_products.csv"
IMAGES_DIR = PROJECT_ROOT / "images"

print("CSV:", CSV_PATH)
print("Images dir:", IMAGES_DIR)

# Load full CSV (all products)
raw_df = pd.read_csv(CSV_PATH)

# Filtered view: only products that have an image file
df = raw_df[raw_df["image_filename"].notna() & (raw_df["image_filename"] != "")]
df = df.reset_index(drop=True)

print("Full CSV rows:", len(raw_df))
print("Filtered rows with images:", len(df))
df.head(3)


# %%
df = pd.read_csv(CSV_PATH)

# Adjust the column name if yours is different
IMAGE_COL = "image_filename"

if IMAGE_COL not in df.columns:
    raise ValueError(f"{IMAGE_COL!r} column not found in CSV. Check your column names.")

# Keep only rows that actually have an image file
df = df[df[IMAGE_COL].notna() & (df[IMAGE_COL] != "")]
df = df.reset_index(drop=True)

print("Rows with images:", len(df))
df.head()


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()


# %%
def embed_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Normalize and flatten to 1D numpy
    emb = outputs[0]
    emb = emb / emb.norm(p=2)
    return emb.cpu().numpy()


# %%
emb_list = []
kept_rows = []

for idx, row in df.iterrows():
    img_name = row[IMAGE_COL]
    img_path = IMAGES_DIR / img_name

    if not img_path.exists():
        # Skip rows with missing files
        continue

    emb = embed_image(img_path)
    emb_list.append(emb)
    kept_rows.append(row)

len(emb_list)


# %%
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm.auto import tqdm

# --- paths ---
PROJECT_ROOT = Path("..").resolve()
CSV_PATH = PROJECT_ROOT / "data" / "san_leandro_products.csv"
IMAGES_DIR = PROJECT_ROOT / "images"

# --- load metadata ---
df = pd.read_csv(CSV_PATH)
df = df[df["image_filename"].notna() & (df["image_filename"] != "")].reset_index(drop=True)
print("rows with images:", len(df))

# ---------- STEP 1: material grouping ----------
MATERIAL_SOURCE_COLS = ["body", "material", "category_slug"]

def infer_material_source_col(df):
    for col in MATERIAL_SOURCE_COLS:
        if col in df.columns:
            return col
    return None

MATERIAL_COL = infer_material_source_col(df)
print("Using material source column:", MATERIAL_COL)

def normalize_material(text: str) -> str:
    t = str(text).lower()

    # --- Tile families ---
    if "porcelain" in t:
        if "wood" in t:
            return "wood_look_porcelain"
        return "porcelain"
    if "ceramic" in t:
        return "ceramic"

    # --- Wood / wood-like families ---
    if "laminate" in t:
        return "laminate"
    if "vinyl" in t or "lvp" in t or "lvt" in t:
        return "vinyl"
    if "engineered" in t:
        return "engineered_wood"
    if "solid" in t and ("hardwood" in t or "wood" in t):
        return "solid_wood"
    if "hardwood" in t or "wood" in t:
        return "wood"

    return "other"

if MATERIAL_COL is not None:
    df["material_group"] = df[MATERIAL_COL].apply(normalize_material)
else:
    df["material_group"] = "other"

print(df["material_group"].value_counts())

# ---------- CLIP model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)

# first time this line runs, it may download the processor files
processor = CLIPProcessor.from_pretrained(model_name)

model.eval()

# --- helpers ---
def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    return img

def embed_images(image_paths, batch_size=16):
    all_embs = []

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        images = [load_image(p) for p in batch_paths]

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)  # [B, D]

        # normalize
        embs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        all_embs.append(embs.cpu().numpy())

    return np.vstack(all_embs)

# --- build list of image paths ---
image_paths = [IMAGES_DIR / fname for fname in df["image_filename"].tolist()]
missing = [p for p in image_paths if not p.exists()]
print("missing image files:", len(missing))

# --- actually compute embeddings ---
image_embs = embed_images(image_paths, batch_size=16)
print("image_embs shape:", image_embs.shape)


# %%
from sklearn.neighbors import NearestNeighbors

# NearestNeighbors with cosine distance (1 - cosine similarity)
nn = NearestNeighbors(n_neighbors=5, metric="cosine")
nn.fit(image_embs)

print("Index built over", image_embs.shape[0], "products")


# %%
def search_similar_by_index(query_idx, top_k=5):
    """
    Return the top_k nearest neighbors for a given product index,
    excluding the product itself, deduping by image, and enforcing
    material-aware filtering.

    - Never return the exact same SKU as the query.
    - Never show the same image twice (even if SKU differs).
    - If the query is any kind of wood/laminate/vinyl/etc, only return
      the *same* material_group (so wood won't match laminate/vinyl/
      solid/engineered and vice versa).
    - Tile vs non-tile is also separated (porcelain/ceramic vs wood stuff).
    """
    # Embedding for the query product
    query_emb = image_embs[query_idx].reshape(1, -1)
    query_row = df.iloc[query_idx]
    query_sku = query_row["sku"]
    query_group = query_row.get("material_group", "other")

    # Ask for some extra neighbors to survive filtering/dedup
    n_neighbors = min(top_k + 20, len(df))
    distances, indices = nn.kneighbors(query_emb, n_neighbors=n_neighbors)
    distances = distances[0]
    indices = indices[0]

    results = []
    seen_images = set()

    # Groups that we treat as wood-family, but we still
    # don't mix them with each other unless group matches exactly.
    wood_groups = {"wood", "engineered_wood", "solid_wood", "laminate", "vinyl"}

    # Tile-like groups
    tile_groups = {"porcelain", "ceramic", "wood_look_porcelain"}

    for dist, idx in zip(distances, indices):
        # Skip the exact same row
        if idx == query_idx:
            continue

        row = df.iloc[idx]

        # Skip same SKU as the query
        if row["sku"] == query_sku:
            continue

        img_key = row["image_filename"]

        # Skip duplicate images
        if img_key in seen_images:
            continue

        candidate_group = row.get("material_group", "other")

        # ---------- MATERIAL FILTERING ----------

        if query_group in wood_groups:
            # If query is wood-like (wood, laminate, vinyl, engineeered, solid),
            # require exact same material_group.
            #
            # This is where your rule kicks in:
            # "if it's wood, exclude laminate vs vinyl vs solid vs engineered"
            # because those are all separate groups.
            if candidate_group != query_group:
                continue
        else:
            # For non-wood queries, keep tile vs non-tile separated.
            query_is_tile = query_group in tile_groups
            candidate_is_tile = candidate_group in tile_groups

            # Don't mix tile with non-tile
            if query_is_tile != candidate_is_tile:
                continue

        # ---------- END MATERIAL FILTERING ----------

        seen_images.add(img_key)

        results.append(
            {
                "rank": len(results) + 1,
                "sku": row["sku"],
                "name": row["name"],
                "category": row["category_slug"],
                "material_group": candidate_group,
                "distance": float(dist),
                "image_path": str(IMAGES_DIR / row["image_filename"]),
            }
        )

        if len(results) >= top_k:
            break

    return results


# %%
from IPython.display import display

# Pick a sample index (try different numbers later)
query_idx = 0

query_row = df.iloc[query_idx]
query_img = load_image(IMAGES_DIR / query_row["image_filename"])

print("Query SKU:", query_row["sku"])
print("Name:", query_row["name"])
print("Category:", query_row["category_slug"])
display(query_img)

results = search_similar_by_index(query_idx, top_k=5)
results


# %%
for r in results:
    print(f"Rank {r['rank']} | SKU {r['sku']} | dist={r['distance']:.4f}")
    display(load_image(Path(r["image_path"])))


# %%
from sklearn.cluster import KMeans

k = 20  # example
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(image_embs)


# %%
EMBS_PATH = PROJECT_ROOT / "data" / "image_embs.npy"
np.save(EMBS_PATH, image_embs)
print("Saved embeddings to:", EMBS_PATH)

# %%
from sklearn.neighbors import NearestNeighbors

# Build kNN index over all embeddings
# (you can tune n_neighbors here; we still cap with top_k later)
nn = NearestNeighbors(n_neighbors=50, metric="cosine")
nn.fit(image_embs)

def search_similar_by_index(query_idx: int, top_k: int = 5):
    """
    Core search: given a row index into df/image_embs, return the top_k
    most similar products (excluding the query itself and same SKU).
    """
    n = len(df)
    if query_idx < 0 or query_idx >= n:
        raise IndexError(f"query_idx {query_idx} out of range for df with {n} rows")

    # Get query embedding
    query_emb = image_embs[query_idx]

    # Ask for top_k + 1 so we can drop the self-match
    distances, indices = nn.kneighbors([query_emb], n_neighbors=top_k + 1)

    distances = distances[0]
    indices = indices[0]

    results = []
    query_row = df.iloc[query_idx]
    query_sku = str(query_row["sku"])

    for idx, dist in zip(indices, distances):
        # Skip the query itself (same index)
        if idx == query_idx:
            continue

        row = df.iloc[idx]
        # Also skip any product with the same SKU (duplicate images, variants)
        if str(row["sku"]) == query_sku:
            continue

        results.append(
            {
                "rank": len(results) + 1,
                "distance": float(dist),
                "sku": str(row["sku"]),
                "name": row["name"],
                "category_slug": row.get("category_slug", ""),
                "image_path": str(IMAGES_DIR / row["image_filename"]),
            }
        )

        if len(results) >= top_k:
            break

    return results


# %%
def get_index_by_sku(query_sku: str) -> int:
    """
    Find the dataframe index for a given SKU in the filtered df
    (the one that has embeddings).

    If it's only in raw_df, raise a clearer error.
    """
    sku_str = str(query_sku).strip()

    # Compare as stripped strings to avoid whitespace issues
    mask = df["sku"].astype(str).str.strip() == sku_str
    matches = df.index[mask].tolist()

    if not matches:
        # If we kept a full copy of the CSV, check there for a better error
        if "raw_df" in globals():
            raw_mask = raw_df["sku"].astype(str).str.strip() == sku_str
            if raw_mask.any():
                raise ValueError(
                    f"SKU {sku_str!r} exists in san_leandro_products.csv "
                    f"but was filtered out of df (likely missing image_filename, "
                    f"so there is no precomputed embedding for it)."
                )

        # Truly not found anywhere
        raise ValueError(f"SKU {sku_str!r} not found in CSV/df.")

    return matches[0]



def get_index_by_name(query_name: str, exact: bool = False, occurrence: int = 0) -> int:
    """
    Find the dataframe index for a given product name.

    - If exact=False (default), uses case-insensitive substring match.
    - If exact=True, requires full-case-insensitive equality.
    - If multiple matches, pick which one with `occurrence` (0 = first).

    Raises ValueError if nothing matches or occurrence is out of range.
    """
    names = df["name"].astype(str)

    if exact:
        mask = names.str.casefold() == query_name.casefold()
    else:
        mask = names.str.contains(query_name, case=False, na=False)

    matches = df.index[mask].tolist()
    if not matches:
        raise ValueError(f"No product name matching {query_name!r} found.")

    if not (0 <= occurrence < len(matches)):
        raise ValueError(
            f"Found {len(matches)} matches for {query_name!r}, "
            f"but occurrence={occurrence} is out of range."
        )

    return matches[occurrence]


def search_similar_by_sku(query_sku, top_k: int = 5):
    """
    Wrapper around search_similar_by_index that lets you search by SKU.
    """
    idx = get_index_by_sku(query_sku)
    return search_similar_by_index(idx, top_k=top_k)


def search_similar_by_name(
    query_name: str,
    top_k: int = 5,
    exact: bool = False,
    occurrence: int = 0,
):
    """
    Wrapper around search_similar_by_index that lets you search by name
    (exact or substring), picking the Nth match if there are multiple.
    """
    idx = get_index_by_name(query_name, exact=exact, occurrence=occurrence)
    return search_similar_by_index(idx, top_k=top_k)


# %%
from IPython.display import display

# --- EXAMPLE: query by SKU ---
# Put a real SKU from your CSV here:
query_sku = "101156321"  # change this to whatever you want

# Find the index and show the query product
query_idx = get_index_by_sku(query_sku)
query_row = df.iloc[query_idx]

print("QUERY PRODUCT")
print("-------------")
print("SKU:", query_row["sku"])
print("Name:", query_row["name"])
print("Category:", query_row.get("category_slug", ""))

# Show query image
query_img_path = IMAGES_DIR / query_row["image_filename"]
try:
    from PIL import Image
    display(Image.open(query_img_path))
except FileNotFoundError:
    print("(Query image file not found at:", query_img_path, ")")
except Exception as e:
    print("(Error opening query image:", e, ")")

# Run similarity search
results = search_similar_by_sku(query_sku, top_k=5)

print("\nSIMILAR PRODUCTS")
print("----------------")
for r in results:
    print(f"Rank {r['rank']} | SKU {r['sku']} | dist={r['distance']:.4f}")
    print("  Name:", r["name"])
    print("  Category:", r["category_slug"])
    try:
        img = Image.open(r["image_path"])
        display(img)
    except FileNotFoundError:
        print("  (Image file not found at:", r['image_path'], ")")
    except Exception as e:
        print("  (Error opening image:", e, ")")


# %%
# --- EXAMPLE: query by product name (substring) ---

# This can be something like "oak", "montauk", "ledger", etc.
query_name = "oak"   # change this as needed

results_by_name = search_similar_by_name(
    query_name,
    top_k=5,
    exact=False,      # set True if you want exact name match
    occurrence=0,     # if multiple matches, which one to use (0 = first)
)

print(f"Query name substring: {query_name!r}")
print("------------------------------")
for r in results_by_name:
    print(f"Rank {r['rank']} | SKU {r['sku']} | dist={r['distance']:.4f}")
    print("  Name:", r["name"])
    print("  Category:", r["category_slug"])
    print()




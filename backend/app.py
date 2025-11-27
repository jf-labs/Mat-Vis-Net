# backend/app.py
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .model import init_model, similar_by_sku, IMAGES_DIR

app = FastAPI(
    title="Mat-Vis-Net API",
    description="SKU similarity search for Floor & Decor San Leandro",
)

# CORS so your web/mobile app can hit it during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve product images so the UI can show them
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


class Product(BaseModel):
    sku: str
    name: str
    category_slug: str | None = None
    material_bucket: str | None = None
    image_filename: str | None = None
    distance: float | None = None
    image_url: str | None = None


class SimilarResponse(BaseModel):
    query: Product
    results: List[Product]


@app.on_event("startup")
def on_startup() -> None:
    init_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/similar-by-sku", response_model=SimilarResponse)
def similar_by_sku_endpoint(sku: str, k: int = 5):
    try:
        payload = similar_by_sku(sku, top_k=k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Add image URL for convenience
    def with_image_url(p: dict) -> dict:
        img = p.get("image_filename")
        return {
            **p,
            "image_url": f"/images/{img}" if img else None,
        }

    query = with_image_url(payload["query"])
    results = [with_image_url(r) for r in payload["results"]]

    return {"query": query, "results": results}

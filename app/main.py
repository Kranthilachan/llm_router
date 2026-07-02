"""
ATLAS Router service entrypoint.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as atlas_router
from app.config import ROUTER_VERSION

app = FastAPI(
    title="ATLAS Router",
    description=(
        "Adaptive Task and LLM Allocation System — a provider-agnostic LLM "
        "routing control plane. Decides which model or execution plan should "
        "handle a request under real operational constraints (governance, "
        "cost, latency, uncertainty, SLAs)."
    ),
    version=ROUTER_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(atlas_router, tags=["atlas-router"])


@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "ATLAS Router",
        "version": ROUTER_VERSION,
        "docs": "/docs",
    }

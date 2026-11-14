"""RetailIQ REST API.   Run: uvicorn api.main:app --reload     Docs: http://localhost:8000/docs"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api import deps
from api.routes import analytics, customers, health, recommendations
from src.utils.config import get_logger

VERSION = "1.0.0"
log = get_logger("retailiq.api")

DESCRIPTION = """
Customer segmentation, value prediction and product recommendations for retail transactions.

* **Privacy**: only aggregated, anonymised data is returned. No names, addresses or individual order lines.
* **Auth**: if the `RETAILIQ_API_KEY` environment variable is set, send it in the `X-API-Key` header.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        deps.get_store()
    except Exception as exc:
        log.warning("Starting without artefacts: %s", getattr(exc, "detail", exc))
    yield


app = FastAPI(title="RetailIQ API", version=VERSION, description=DESCRIPTION, lifespan=lifespan,
              openapi_tags=[{"name": "System"}, {"name": "Customers"}, {"name": "Recommendations"}, {"name": "Analytics"}])
for r in (health.router, customers.router, recommendations.router, analytics.router):
    app.include_router(r)


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid, t = uuid.uuid4().hex[:8], time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("[%s] unhandled error on %s %s", rid, request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})
    log.info("[%s] %s %s -> %d (%.0f ms)", rid, request.method, request.url.path, response.status_code, (time.perf_counter() - t) * 1000)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    msgs = []
    for e in exc.errors():
        loc = ".".join(str(p) for p in e["loc"] if p != "body")
        msg = e["msg"].removeprefix("Value error, ")
        msgs.append(f"{loc}: {msg}" if loc else msg)
    return JSONResponse(status_code=422, content={"detail": "; ".join(msgs)})

from fastapi import APIRouter

from api import deps
from api.schemas.models import HealthResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health() -> HealthResponse:
    from api.main import VERSION
    try:
        s = deps.get_store()
    except Exception as exc:  # artefacts not built yet -> report, do not crash
        return HealthResponse(status="degraded", version=VERSION, artifacts_loaded=False, detail=getattr(exc, "detail", str(exc)))
    return HealthResponse(status="ok", version=VERSION, artifacts_loaded=True, data_source=s.meta["data_source"],
                          customers=len(s.scores), selected_model=s.meta["selected_model"])

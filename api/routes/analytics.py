from fastapi import APIRouter, Depends

from api import deps
from api.schemas.models import SegmentSummary, SegmentsAnalytics
from src.models.customer_segmentation import SEGMENT_PLAYBOOK

router = APIRouter(prefix="/analytics", tags=["Analytics"], dependencies=[Depends(deps.require_api_key)])


@router.get("/segments", response_model=SegmentsAnalytics, summary="Aggregated segment statistics (no customer-level data)")
def segments(store: deps.ArtifactStore = Depends(deps.get_store)) -> SegmentsAnalytics:
    rows = []
    for name, r in store.profile.iterrows():
        play = SEGMENT_PLAYBOOK.get(name.replace(" II", ""), {"who": "", "action": ""})
        rows.append(SegmentSummary(segment=name, customers=int(r.customers), customer_share=round(float(r.customer_share), 4),
                                   revenue_share=round(float(r.revenue_share), 4), median_recency_days=float(r.recency_median),
                                   median_frequency=float(r.frequency_median), median_monetary=round(float(r.monetary_median), 2),
                                   who=play["who"], recommended_action=play["action"]))
    return SegmentsAnalytics(n_segments=len(rows), silhouette=round(store.meta["silhouette"], 4), segments=rows)

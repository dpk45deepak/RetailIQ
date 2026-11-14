import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, status

from api import deps
from api.schemas.models import (CustomerProfile, ErrorResponse, PredictValueRequest, PredictValueResponse, RFM, SegmentResponse)
from src.models.customer_segmentation import SEGMENT_PLAYBOOK

router = APIRouter(prefix="/customers", tags=["Customers"], dependencies=[Depends(deps.require_api_key)])
CustomerId = Path(..., ge=1, le=999_999, description="Anonymised customer id")
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Customer not found"}}


def _row(store: deps.ArtifactStore, customer_id: int) -> pd.Series:
    if customer_id not in store.scores.index:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found.")
    return store.scores.loc[customer_id]


@router.get("/{customer_id}", response_model=CustomerProfile, responses=NOT_FOUND, summary="Aggregated customer profile")
def get_customer(customer_id: int = CustomerId, store: deps.ArtifactStore = Depends(deps.get_store)) -> CustomerProfile:
    r = _row(store, customer_id)
    return CustomerProfile(customer_id=customer_id, segment=r["segment"], country=r["country"],
                           rfm=RFM(recency_days=int(r["recency_days"]), frequency=int(r["frequency"]), monetary=round(float(r["monetary"]), 2)),
                           avg_order_value=round(float(r["avg_order_value"]), 2), active_months=int(r["active_months"]),
                           tenure_days=int(r["tenure_days"]), p_high_value=round(float(r["p_high_value"]), 4))


@router.get("/{customer_id}/segment", response_model=SegmentResponse, responses=NOT_FOUND, summary="Customer segment and playbook")
def get_segment(customer_id: int = CustomerId, store: deps.ArtifactStore = Depends(deps.get_store)) -> SegmentResponse:
    seg = _row(store, customer_id)["segment"]
    play = SEGMENT_PLAYBOOK.get(seg.replace(" II", ""), {"who": "", "action": ""})
    return SegmentResponse(customer_id=customer_id, segment=seg, description=play["who"], recommended_action=play["action"])


@router.post("/{customer_id}/predict-value", response_model=PredictValueResponse, responses=NOT_FOUND,
             summary="Probability of being a high-value customer over the next 90 days")
def predict_value(body: PredictValueRequest | None = None, customer_id: int = CustomerId,
                  store: deps.ArtifactStore = Depends(deps.get_store)) -> PredictValueResponse:
    r = _row(store, customer_id)
    bundle = store.value_bundle
    x = pd.DataFrame([r[bundle["features"]].astype(float)])
    base = float(bundle["pipeline"].predict_proba(x)[0, 1])
    overrides = (body.overrides if body else {}) or {}
    for k, v in overrides.items():
        x[k] = v
    p = float(bundle["pipeline"].predict_proba(x)[0, 1])
    return PredictValueResponse(customer_id=customer_id, probability_high_value=round(p, 4), predicted_high_value=p >= bundle["threshold"],
                                decision_threshold=bundle["threshold"], horizon_days=bundle["horizon_days"], model=bundle["model_name"],
                                baseline_probability=round(base, 4) if overrides else None)

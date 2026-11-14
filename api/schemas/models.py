"""Pydantic request/response models (these also generate the Swagger docs)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.features.customer_features import FEATURE_COLUMNS


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    artifacts_loaded: bool
    data_source: str | None = Field(None, description="'synthetic' means demo data, not the real UCI dataset")
    customers: int | None = None
    selected_model: str | None = None
    detail: str | None = None


class RFM(BaseModel):
    recency_days: int = Field(..., description="Days since last purchase")
    frequency: int = Field(..., description="Number of orders")
    monetary: float = Field(..., description="Net spend in GBP")


class CustomerProfile(BaseModel):
    """Aggregated, non-identifying profile. No names, addresses or order lines are ever exposed."""
    customer_id: int
    segment: str
    country: str
    rfm: RFM
    avg_order_value: float
    active_months: int
    tenure_days: int
    p_high_value: float = Field(..., ge=0, le=1)


class SegmentResponse(BaseModel):
    customer_id: int
    segment: str
    description: str
    recommended_action: str


class PredictValueRequest(BaseModel):
    """Optional what-if: override some features to see how the probability would change."""
    overrides: dict[str, float] = Field(default_factory=dict, examples=[{"recency_days": 10}])

    @field_validator("overrides")
    @classmethod
    def _known_features(cls, v: dict[str, float]) -> dict[str, float]:
        bad = sorted(set(v) - set(FEATURE_COLUMNS))
        if bad:
            raise ValueError(f"unknown feature(s) {bad}; allowed: {FEATURE_COLUMNS}")
        return v


class PredictValueResponse(BaseModel):
    customer_id: int
    probability_high_value: float = Field(..., ge=0, le=1)
    predicted_high_value: bool
    decision_threshold: float
    horizon_days: int
    model: str
    baseline_probability: float | None = Field(None, description="Probability without overrides (only when overrides are given)")


class RecommendationRequest(BaseModel):
    customer_id: int | None = Field(None, ge=1, le=999_999)
    basket: list[str] | None = Field(None, max_length=50, description="StockCodes, for anonymous visitors")
    top_n: int = Field(5, ge=1, le=20)
    strategy: Literal["hybrid", "collaborative", "content", "popularity"] = "hybrid"
    model_config = ConfigDict(json_schema_extra={"example": {"customer_id": 12347, "top_n": 5, "strategy": "hybrid"}})

    @model_validator(mode="after")
    def _need_identity(self):
        if self.customer_id is None and not self.basket:
            raise ValueError("provide customer_id or a non-empty basket")
        return self


class RecommendedItem(BaseModel):
    stock_code: str
    description: str
    score: float
    collaborative: float
    content: float
    popularity: float
    reason: str


class RecommendationResponse(BaseModel):
    customer_id: int | None
    strategy_used: str
    fallback: bool = Field(..., description="True when there was no history and best-sellers were returned")
    items: list[RecommendedItem]


class SegmentSummary(BaseModel):
    segment: str
    customers: int
    customer_share: float
    revenue_share: float
    median_recency_days: float
    median_frequency: float
    median_monetary: float
    who: str
    recommended_action: str


class SegmentsAnalytics(BaseModel):
    n_segments: int
    silhouette: float
    segments: list[SegmentSummary]


class ErrorResponse(BaseModel):
    detail: str

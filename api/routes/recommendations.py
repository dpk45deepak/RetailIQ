from fastapi import APIRouter, Depends

from api import deps
from api.schemas.models import RecommendationRequest, RecommendationResponse, RecommendedItem

router = APIRouter(tags=["Recommendations"], dependencies=[Depends(deps.require_api_key)])


@router.post("/recommendations", response_model=RecommendationResponse, summary="Top-N product recommendations",
             description="Unknown customers with no basket receive best-sellers (`fallback=true`) instead of an error.")
def recommend(body: RecommendationRequest, store: deps.ArtifactStore = Depends(deps.get_store)) -> RecommendationResponse:
    df = store.recommender.recommend(customer_id=body.customer_id, n=body.top_n, strategy=body.strategy, basket=body.basket)
    fallback = bool(df["fallback"].iloc[0]) if len(df) else True
    used = "popularity" if fallback else body.strategy
    items = [RecommendedItem(**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "fallback"})
             for row in df.to_dict("records")]
    return RecommendationResponse(customer_id=body.customer_id, strategy_used=used, fallback=fallback, items=items)

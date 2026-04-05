from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from anima_nasdetr.infer import run_infer


class PredictRequest(BaseModel):
    image: str
    variant: str = "A1"
    num_queries: int | None = None


app = FastAPI(title="DEF-nasdetr API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        return run_infer(req.image, req.variant, req.num_queries)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"prediction_failed: {exc}") from exc

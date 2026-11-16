from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request

from .schemas import Applicant, ExplanationResponse, PredictionResponse
from .service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ModelService()   # loaded once, not on every request
    yield


app = FastAPI(title="Credit Scoring API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health(request: Request):
    svc: ModelService = request.app.state.service
    return {"status": "ok", "model_version": svc.version, "threshold": svc.threshold}


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: Applicant, request: Request):
    svc: ModelService = request.app.state.service
    return svc.predict(applicant.to_frame())


@app.post("/explain", response_model=ExplanationResponse)
def explain(applicant: Applicant, request: Request, top_k: int = Query(5, ge=1, le=20)):
    svc: ModelService = request.app.state.service
    df = applicant.to_frame()
    return {**svc.predict(df), "top_contributions": svc.explain(df, top_k)}
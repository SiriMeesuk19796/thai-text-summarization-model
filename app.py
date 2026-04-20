from fastapi import FastAPI
from pydantic import BaseModel
from bert_score import score
from enum import Enum
from services.mt5_service import summarize_mt5
from services.gemini_service import summarize_gemini

app = FastAPI()

class ModelName(str, Enum):
    mt5 = "mt5"
    gemini = "gemini"

class Request(BaseModel):
    text: str
    mode: str = "normal"
    reference: str = None
    model: ModelName


@app.post("/summarize/mt5")
def summarize_mt5_api(req: Request):
    return {"summary": summarize_mt5(req.text, req.mode)}


@app.post("/summarize/gemini")
def summarize_gemini_api(req: Request):
    try:
        summary = summarize_gemini(req.text, req.mode)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}


@app.post("/evaluate")
def evaluate(req: Request):
    
    if req.model == "mt5":
        summary = summarize_mt5(req.text, req.mode)
    elif req.model == "gemini":
        summary = summarize_gemini(req.text, req.mode)
    else:
        return {"error": "invalid model"}

    if not req.reference:
        return {"summary": summary, "bertscore": None}

    P, R, F1 = score([summary], [req.reference], lang="th")

    return {
        "summary": summary,
        "bertscore": {
            "precision": float(P[0]),
            "recall": float(R[0]),
            "f1": float(F1[0])
        }
    }


# 🔥 compare 2 model (โคตร useful)
@app.post("/compare")
def compare(req: Request):
    mt5_sum = summarize_mt5(req.text, req.mode)
    gemini_sum = summarize_gemini(req.text, req.mode)

    result = {
        "mt5": mt5_sum,
        "gemini": gemini_sum
    }

    if req.reference:
        P1, R1, F1 = score([mt5_sum], [req.reference], lang="th")
        P2, R2, F2 = score([gemini_sum], [req.reference], lang="th")

        result["score"] = {
            "mt5": float(F1[0]),
            "gemini": float(F2[0])
        }

    return result
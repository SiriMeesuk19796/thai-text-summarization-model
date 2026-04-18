from fastapi import FastAPI
from pydantic import BaseModel
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bert_score import score

app = FastAPI()

model_name = "thanathorn/mt5-cpe-kmutt-thai-sentence-sum"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

WHITESPACE_HANDLER = lambda x: re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', x.strip()))

class Request(BaseModel):
    text: str
    mode: str = "normal"
    reference: str = None


# ✅ summarize
@app.post("/summarize")
def summarize(req: Request):

    if not req.text.strip():
        return {"error": "text is empty"}

    clean_text = WHITESPACE_HANDLER(req.text)

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="longest"
    ).to(device)

    if req.mode == "teaser":
        min_len = 10
        max_len = 30
    elif req.mode == "short":
        min_len = 20
        max_len = 60
    else:
        min_len = 50
        max_len = 120

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"summary": summary}


@app.post("/evaluate")
def evaluate(req: Request):

    if not req.text.strip():
        return {"error": "text is empty"}

    clean_text = WHITESPACE_HANDLER(req.text)

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="longest"
    ).to(device)

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=120,
        min_length=50,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if not req.reference:
        return {
            "summary": summary,
            "bertscore": None
        }

    P, R, F1 = score(
        [summary],
        [req.reference],
        lang="th"
    )

    return {
        "summary": summary,
        "bertscore": {
            "precision": float(P[0]),
            "recall": float(R[0]),
            "f1": float(F1[0])
        }
    }
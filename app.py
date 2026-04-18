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


# ✅ helper: mode → length
def get_length_by_mode(mode: str):
    if mode == "teaser":
        return 10, 30
    elif mode == "short":
        return 20, 60
    else:
        return 50, 120


# ✅ helper: generate summary (ใช้ร่วมกัน)
def generate_summary(text: str, mode: str):
    clean_text = WHITESPACE_HANDLER(text)

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="longest"
    ).to(device)

    min_len, max_len = get_length_by_mode(mode)

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

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ✅ summarize
@app.post("/summarize")
def summarize(req: Request):

    if not req.text.strip():
        return {"error": "text is empty"}

    summary = generate_summary(req.text, req.mode)

    return {"summary": summary}


# ✅ evaluate (แก้ให้ใช้ mode แล้ว)
@app.post("/evaluate")
def evaluate(req: Request):

    if not req.text.strip():
        return {"error": "text is empty"}

    # 🔥 ใช้ mode แล้ว
    summary = generate_summary(req.text, req.mode)

    if not req.reference:
        return {
            "summary": summary,
            "bertscore": None
        }

    # 🔥 BERTScore
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
import os
from dotenv import load_dotenv
from fastapi import FastAPI
import google.generativeai as genai
from bert_score import score
from text_cleaner import clean_text
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("models/gemini-flash-lite-latest")

import google.generativeai as genai

for m in genai.list_models():
    print(m.name)

class Request(BaseModel):
    text: str
    mode: str = "normal"
    reference: str = None

@app.post("/summarize")
def summarize(req: Request):

    if not req.text.strip():
        return {"error": "text is empty"}

    summary = generate_summary(req.text, req.mode)

    return {"summary": summary}

@app.post("/evaluate")
def evaluate(req: Request):

    if not req.text.strip():
        return {"error": "text is empty"}

    summary = generate_summary(req.text, req.mode)

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


def get_length_by_mode(mode: str):
    if mode == "teaser":
        return 10, 30
    elif mode == "short":
        return 20, 60
    else:
        return 50, 120


def generate_summary(text: str, mode: str):
    clean = clean_text(text)
    min_len, max_len = get_length_by_mode(mode)

    prompt = f"""
    สรุปข่าวต่อไปนี้เป็นภาษาไทย:
    - ต้องมีความยาวระหว่าง {min_len} ถึง {max_len} คำเท่านั้น
    - ห้ามสั้นกว่านี้โดยเด็ดขาด
    - ห้ามจบกลางประโยค
    - ต้องเขียนให้จบครบสมบูรณ์
    - กระชับ เข้าใจง่าย

    เนื้อหา:
    {clean}
    """

    generation_config={
        "max_output_tokens": 300,
        "temperature": 0.3,  
        "top_p": 0.9
    }

    response = model.generate_content(prompt)

    return response.text.strip()
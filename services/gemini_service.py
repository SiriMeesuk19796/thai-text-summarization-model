import os
from dotenv import load_dotenv
import google.generativeai as genai
from utils.text_cleaner import clean_text

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("models/gemini-2.5-flash")

import google.generativeai as genai

for m in genai.list_models():
    print(m.name)


def get_length_by_mode(mode: str):
    if mode == "teaser":
        return 10, 30
    elif mode == "short":
        return 20, 60
    else:
        return 50, 120


def summarize_gemini(text: str, mode: str):
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
        "temperature": 0.5,  
        "top_p": 0.9
    }

    response = model.generate_content(prompt)

    return response.text.strip()
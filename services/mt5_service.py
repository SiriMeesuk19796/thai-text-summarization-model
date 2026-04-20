import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.text_cleaner import clean_text

model_name = "thanathorn/mt5-cpe-kmutt-thai-sentence-sum"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def get_length_by_mode(mode: str):
    if mode == "teaser":
        return 10, 30
    elif mode == "short":
        return 20, 60
    else:
        return 50, 120


def summarize_mt5(text: str, mode: str):
    clean = clean_text(text)

    inputs = tokenizer(
        clean,
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
FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
model_name='thanathorn/mt5-cpe-kmutt-thai-sentence-sum'; \
AutoTokenizer.from_pretrained(model_name); \
AutoModelForSeq2SeqLM.from_pretrained(model_name)"

COPY . .

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
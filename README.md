# Thai Text Summarization — Model API

Thai Lizard model server — Python FastAPI service for Thai text summarization using Google Gemini.  
Built as part of CP465 Text Mining (2/2568).

---

## Tech Stack

| Category | Library |
|----------|---------|
| Framework | FastAPI |
| Server | Uvicorn |
| AI Model | Google Gemini (`google-generativeai`) |
| Evaluation | BERTScore + PyTorch |
| Text Processing | transformers |
| Config | python-dotenv |
| Containerization | Docker |

---

## Project Structure
```
├── app.py              # FastAPI app
├── text_cleaner.py     # Thai text preprocessing
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python >= 3.10
- pip
- Docker (optional)

### Installation

```bash
git clone https://github.com/SiriMeesuk19796/thai-text-summarization-model.git
cd thai-text-summarization-model
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file at the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### Run Development Server

```bash
uvicorn app:app --reload --port 8000
```

API will be available at `http://localhost:8000`

### Run with Docker

```bash
docker build -t thai-lizard-model .
docker run -p 8000:8000 --env-file .env thai-lizard-model
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/summarize` | Summarize Thai text using Gemini |

### Request Body

```json
{
  "text": "ข้อความที่ต้องการสรุป",
  "mode": "short",
  "reference": "optional reference text for BERTScore"
}
```

### Available Modes

| Mode | Description |
|------|-------------|
| `teaser` | Short teaser / headline |
| `short` | Brief summary (≤3 lines) |
| `normal` | Standard summary (≤8 lines) |

### Response

```json
{
  "summary": "ข้อความที่สรุปแล้ว",
  "original_text": "ข้อความต้นฉบับ",
  "frontend_metric": {
    "score": 85
  }
}
```

---

## Text Preprocessing

`text_cleaner.py` provides a `clean_text()` function that:

- Strips leading/trailing whitespace
- Removes multiple newlines
- Removes multiple consecutive spaces
- Filters out non-Thai/non-ASCII unicode characters

---

## Team

| Name | Student ID |
|------|-----------|
| Rujapa Monkhontirapat | 65102010201 |
| Siri Meesuk | 65102010202 |
| Thatchaya Siriwaseree | 65102010417 |

---

## Related Repositories

- [Frontend](https://github.com/rujapathz/thai-text-summarization-frontend)
- [Backend](https://github.com/rujapathz/thai-text-summarization-backend)

import re

def clean_text(text: str):
    text = text.strip()

    # remove multiple newlines
    text = re.sub(r'\n+', ' ', text)

    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # remove weird unicode (optional)
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.,!?():/%\-\"\'\n]', '', text)

    return text
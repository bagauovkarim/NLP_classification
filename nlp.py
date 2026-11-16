from bs4 import BeautifulSoup
import re
from collections import Counter

# Функция для чистки текста
def strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text()

# Функция токенизации по словам
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Функция подсчета токенов 
def count_tokens(token_series, stop_words):
    counter = Counter()
    for token_list in token_series:
        clear_tokens = [t for t in token_list if t not in stop_words] # Фильтрация стоп слов
        counter.update(clear_tokens)


    return counter
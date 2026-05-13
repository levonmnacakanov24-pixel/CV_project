import requests
import pandas as pd
import time
import re

langs = ['ru', 'uk', 'be', 'pl', 'cs', 'es', 'pt', 'fr', 'it', 'en', 'de', 'nl']
ARTICLES_PER_LANG = 150
BATCH_SIZE = 10

headers = {"User-Agent": "LanguageDetectionProject/1.0 (student_project@example.com)"}


# Функция для очистки текста
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip().lower()

data = []

# Собираем тексты для каждого языка
for lang in langs:
    print(f"\nСобираем язык: {lang}")
    count = 0
    url = f"https://{lang}.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": str(BATCH_SIZE),
        "prop": "extracts",
        "exchars": "200",
        "explaintext": "1",
    }

    attempts = 0
    while count < ARTICLES_PER_LANG and attempts < 200:
        attempts += 1
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            result = response.json()

            if 'query' in result and 'pages' in result['query']:
                for page_id, page_info in result['query']['pages'].items():
                    if count >= ARTICLES_PER_LANG: break

                    if 'extract' in page_info and page_info['extract']:
                        text = clean_text(page_info['extract'])

                        if len(text) > 100:
                            data.append({'text': text, 'language': lang})
                            count += 1

            print(f"Собрано {count}/{ARTICLES_PER_LANG}")
            time.sleep(0.1)

        except requests.exceptions.Timeout:
            print("Таймаут")
        except Exception as e:
            time.sleep(2)

df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('wikipedia_dataset.csv', index=False, encoding='utf-8')
print(f"Сохранено {len(df)} текстов.")

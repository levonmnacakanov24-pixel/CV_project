import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix
import re


# Функция для очистки текста
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip().lower()


# Функция извлечения n-грамм
def get_ngrams(text, min_n=2, max_n=4):
    text = f" {text} "
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i + n])
    return ngrams


# Строим профили языков
def build_profile(texts, top_k=300):
    ngram_counts = Counter()
    for text in texts:
        ngram_counts.update(get_ngrams(text))

    # most_common возвращает список кортежей: [(' th', 500), ('he ', 450), ...]
    return [ngram for ngram, count in ngram_counts.most_common(top_k)]


# Обучение на полученных данных
df = pd.read_csv('wikipedia_dataset.csv')

# Словарь для хранения профилей всех языков
language_profiles = {}

for lang in df['language'].unique():
    print(f"Строим профиль для языка: {lang}")
    texts = df[df['language'] == lang]['text'].tolist()
    cleaned_texts = [clean_text(t) for t in texts]
    language_profiles[lang] = build_profile(cleaned_texts, top_k=300)


# Функция вычисления расстояния
def calculate_distance(text_profile, lang_profile):
    distance = 0
    max_penalty = len(lang_profile)

    for text_rank, ngram in enumerate(text_profile):
        if ngram in lang_profile:
            # Находим позицию этой n-граммы в профиле языка
            lang_rank = lang_profile.index(ngram)
            distance += abs(text_rank - lang_rank)
        else:
            # Если такой n-граммы в языке нет, даем максимальный штраф
            distance += max_penalty

    return distance


# Функция предсказания
def predict_language(text):
    cleaned = clean_text(text)
    # Строим профиль из текста
    text_profile = build_profile([cleaned], top_k=300)

    best_lang = None
    min_dist = float('inf')

    # Сравниваем текст со всеми языками
    for lang, lang_profile in language_profiles.items():
        dist = calculate_distance(text_profile, lang_profile)
        # Ищем язык с минимальным штрафом
        if dist < min_dist:
            min_dist = dist
            best_lang = lang

    return best_lang

# Еще один датасет для оценки погрешности
bd = pd.read_csv("wikipedia_dataset_2_0.csv")

y_true = bd["language"].tolist()
y_pred = [predict_language(text) for text in bd["text"]]

# Отсортированный список языков
labels = sorted(set(y_true))

# Строим матрицу ошибок
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Матрица ошибок', fontsize=16)
plt.xlabel('Предсказанный язык', fontsize=12)
plt.ylabel('Истинный язык', fontsize=12)

plt.savefig('confusion_matrix_1_0.png', dpi=600, bbox_inches='tight')

plt.show()

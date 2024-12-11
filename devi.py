from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import threading
import time
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import ne_chunk, pos_tag
from nltk.tree import Tree
import string

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Initialize Flask app
app = Flask(__name__)

# Global variables
news_data = {}
model = None
TOKENIZER = None
max_length = 100
similarity_threshold = 0.7

# Website configurations
WEB_CONFIG = {
    "BBC": {
        "url": "https://www.bbc.com/news",
        "content_selector": 'h2[data-testid="card-headline"], p[data-testid="card-description"]',
        "publisher": "BBC"
    },
    "CNN": {
        "url": "https://edition.cnn.com/world",
        "content_selector": 'span.container__headline-text, div.l-container p',
        "publisher": "CNN"
    },
    "The Hindu": {
        "url": "https://www.thehindu.com/",
        "content_selector": 'strong, a.cx-item.cx-main',
        "publisher": "The Hindu"
    },
    "Google News": {
        "url": "https://news.google.com/topstories",
        "content_selector": 'a.gPFEn',
        "publisher": "Google News"
    }
}

# HTML Template for UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Real or Fake Checker</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css">
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center text-primary">News Real or Fake Checker</h1>
        <form method="POST" action="/check">
            <div class="mb-3">
                <label for="news" class="form-label">Enter News Headline</label>
                <input type="text" id="news" name="news" class="form-control" required>
            </div>
            <button type="submit" class="btn btn-success">Check</button>
        </form>
        {% if result %}
        <div class="alert alert-info mt-4" role="alert">
            <strong>Result:</strong> {{ result }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Load the model and tokenizer
def load_dependencies():
    global model, TOKENIZER
    model = load_model("news_classification_model.h5")
    data = pd.read_csv("news_dataset.csv")
    if "Text" not in data.columns:
        raise ValueError("The dataset does not contain a 'Text' column.")

    TOKENIZER = Tokenizer(num_words=5000)
    TOKENIZER.fit_on_texts(data["Text"].values)

# Preprocess text for prediction
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    sequence = TOKENIZER.texts_to_sequences([" ".join(filtered_tokens)])
    return pad_sequences(sequence, maxlen=max_length)

# Predict whether the news is real or fake
def predict_news(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    class_label = prediction.argmax(axis=1)[0]
    return "Real" if class_label == 1 else "Fake"

# Semantic similarity

def semantic_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0]
    return similarity


# Named Entity Extraction
def extract_entities(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    entities = set()
    for chunk in chunked:
        if isinstance(chunk, Tree):
            entity = " ".join(c[0] for c in chunk)
            entities.add(entity.lower())
    return entities

# Entity Comparison
def compare_entities(entities1, entities2):
    if not entities1 or not entities2:
        return 0
    overlap = entities1.intersection(entities2)
    return len(overlap) / max(len(entities1), len(entities2))

# Crawl news for comparison
def crawl_news(website):
    global news_data
    config = WEB_CONFIG[website]

    edge_driver_path = r"C:/webdrivers/msedgedriver.exe"  # Update path if necessary
    options = EdgeOptions()
    options.add_argument("--headless")
    service = EdgeService(executable_path=edge_driver_path)
    driver = webdriver.Edge(service=service, options=options)

    try:
        driver.get(config["url"])
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, config["content_selector"]))
        )
        elements = driver.find_elements(By.CSS_SELECTOR, config["content_selector"])

        data = []
        for idx, element in enumerate(elements):
            content = element.text.strip()
            if content:
                data.append({
                    "S.No": idx + 1,
                    "Content": content,
                    "Publisher": config["publisher"],
                    "Date": time.strftime("%Y-%m-%d")
                })
        news_data[website] = pd.DataFrame(data)
    except Exception as e:
        print(f"Error while crawling {website}: {e}")
    finally:
        driver.quit()

# Background thread for periodic updates
def schedule_updates():
    while True:
        for site in WEB_CONFIG.keys():
            print(f"Starting crawl for {site}...")
            crawl_news(site)
        time.sleep(24 * 3600)

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, result=None)

@app.route("/check", methods=["POST"])
def check_news():
    news = request.form["news"]
    similar_news = None
    highest_similarity = 0

    for site, df in news_data.items():
        for content in df["Content"]:
            # Semantic similarity
            semantic_score = semantic_similarity(news, content)

            # Named entity comparison
            input_entities = extract_entities(news)
            content_entities = extract_entities(content)
            entity_score = compare_entities(input_entities, content_entities)

            # Weighted similarity (Adjust weights as needed)
            total_score = 0.7 * semantic_score + 0.3 * entity_score

            if total_score > highest_similarity:
                highest_similarity = total_score
                similar_news = content

    if highest_similarity >= similarity_threshold and similar_news:
        result = f"Real (Matched with: '{similar_news[:100]}...', Score: {highest_similarity:.2f})"
    else:
        result = predict_news(news)

    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == "__main__":
    load_dependencies()
    threading.Thread(target=schedule_updates, daemon=True).start()
    app.run(debug=True, port=5000)

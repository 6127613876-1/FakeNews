from flask import Flask, request, render_template_string
import praw
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from goose3 import Goose
from urllib.parse import urljoin
import requests

# Initialize Flask app
app = Flask(__name__)

# Reddit API credentials (replace these with your own)
reddit_instance = praw.Reddit(
    client_id="lrdkKHOzPtBVb_0XDp7fqQ",
    client_secret="R0ZWqtoHDS6363i9ETJxP4hkOi0j8Q",
    username="Own_Welcome2864",
    password="Nux%&e_N4Zpg8uy",
    user_agent="test_bot",
)

# News sources configuration
SUBREDDITS = {
    "technology": "Tech",
    "politics": "Politics",
    "sports": "Sports",
}

# Global variable to store news data
news_data = {}

# Scrape Reddit with the provided logic
def scrape_reddit():
    data = []
    s_no = 1
    for subreddit_name, subject in SUBREDDITS.items():
        subreddit = reddit_instance.subreddit(subreddit_name)
        for submission in subreddit.top(limit=25):  # Adjust limit as needed
            data.append({
                "S.No": s_no,
                "Content": submission.title,
                "Publisher": subject,
                "Date": datetime.now().strftime("%Y-%m-%d"),
            })
            s_no += 1
    return pd.DataFrame(data).to_dict(orient="records")

# Scrape The Hindu
def scrape_the_hindu():
    url = 'https://www.thehindu.com/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    final_urls = [link.attrs.get("href") for link in soup.select("a[href$='.ece']")]
    return list(set(final_urls))

# Scrape OpIndia
def scrape_op_india():
    now = datetime.today()
    current_year = str(now.year)
    current_month = f"{now.month:02d}"
    base_url = 'https://www.opindia.com/'
    pattern = f'{base_url}{current_year}/{current_month}/'

    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    final_urls = [
        urljoin(base_url, link.attrs.get("href"))
        for link in soup.find_all('a')
        if pattern in urljoin(base_url, link.attrs.get("href"))
    ]
    return list(set(final_urls))

# Process articles
def process_articles(urls):
    titles, texts, authors = [], [], []
    g = Goose()

    for url in urls:
        try:
            article = g.extract(url=url)
            title = article.title or "No Title"
            text = article.cleaned_text or "No Text"
            domain = article.domain or "Unknown Domain"
            authors.append(domain)
            titles.append(title)
            texts.append(text)
        except Exception as e:
            print(f"Failed to process {url}: {e}")

    return titles, texts, authors

# HTML templates
HTML_TEMPLATE_INDEX = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Scraper</title>
</head>
<body>
    <h1>News Scraper</h1>
    <form action="/scrape" method="post">
        <label for="news_source">Select News Source:</label>
        <select id="news_source" name="news_source">
            <option value="reddit">Reddit</option>
            <option value="hindu">The Hindu</option>
            <option value="opindia">OpIndia</option>
        </select>
        <button type="submit">Scrape</button>
    </form>
</body>
</html>
'''

HTML_TEMPLATE_RESULTS = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scraped News</title>
</head>
<body>
    <h1>Scraped News Data</h1>
    <table border="1">
        <thead>
            <tr>
                <th>S.No</th>
                <th>Content</th>
                <th>Publisher</th>
                <th>Date</th>
            </tr>
        </thead>
        <tbody>
            {{ rows|safe }}
        </tbody>
    </table>
    <a href="/">Back</a>
</body>
</html>
'''

# Flask routes
@app.route('/')
def index():
    return HTML_TEMPLATE_INDEX

@app.route('/scrape', methods=['POST'])
def scrape():
    news_source = request.form['news_source']
    if news_source == 'reddit':
        posts = scrape_reddit()
        rows = ''.join([
            f"<tr><td>{p['S.No']}</td><td>{p['Content']}</td><td>{p['Publisher']}</td><td>{p['Date']}</td></tr>"
            for p in posts
        ])
    elif news_source == 'hindu':
        urls = scrape_the_hindu()
        titles, texts, authors = process_articles(urls)
        rows = ''.join([
            f"<tr><td>{i + 1}</td><td>{t}</td><td>{a}</td><td>{datetime.now().strftime('%Y-%m-%d')}</td></tr>"
            for i, (t, a) in enumerate(zip(titles, authors))
        ])
    elif news_source == 'opindia':
        urls = scrape_op_india()
        titles, texts, authors = process_articles(urls)
        rows = ''.join([
            f"<tr><td>{i + 1}</td><td>{t}</td><td>{a}</td><td>{datetime.now().strftime('%Y-%m-%d')}</td></tr>"
            for i, (t, a) in enumerate(zip(titles, authors))
        ])
    else:
        return "Invalid News Source", 400

    return render_template_string(HTML_TEMPLATE_RESULTS, rows=rows)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)

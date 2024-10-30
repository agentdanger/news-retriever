from flask import Flask, request, make_response
from google.cloud import storage

import asyncio
import aiohttp
import feedparser
import requests
from io import BytesIO
import math
import json
import re
from datetime import datetime, timezone, timedelta
from sentence_transformers import SentenceTransformer
from PIL import Image
from bs4 import BeautifulSoup

app = Flask(__name__)

# Initialize the model globally to avoid loading it multiple times
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define feeds and other constants
feeds = {
    'analytics vidhya': 'https://www.analyticsvidhya.com/feed/',
    'dataquest': 'https://www.dataquest.io/blog/feed/',
    'dataconomy': 'https://dataconomy.com/feed/',
    'machinelearning mastery': 'https://machinelearningmastery.com/feed/',
    'towards data science': 'https://towardsdatascience.com/feed',
    'adexchanger': 'https://adexchanger.com/feed/',
    'adweek': 'https://www.adweek.com/feed/',
    'wsj': 'https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml',
    'nyt': 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'cnn': 'http://rss.cnn.com/rss/cnn_topstories.rss',
    'foxnews': 'http://feeds.foxnews.com/foxnews/latest',
    'nbcnews': 'http://feeds.nbcnews.com/nbcnews/public/news',
    'bbc': 'http://feeds.bbci.co.uk/news/rss.xml',
    'apnews topnews': 'https://rsshub.app/apnews/topics/ap-top-news',
    'apnews tech': 'https://rsshub.app/apnews/topics/technology',
    'bloomberg': 'https://feeds.bloomberg.com/technology/news.rss',
    'business insider': 'https://markets.businessinsider.com/rss/news',
    'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
    'forbes': 'https://www.forbes.com/real-time/feed2/',
    'fortune': 'https://fortune.com/feed/',
    'npr': 'https://feeds.npr.org/1019/rss.xml',
    'marketwatch': 'https://www.marketwatch.com/rss/topstories',
    'money': 'https://money.com/feed/',
    'arxiv': 'http://export.arxiv.org/rss/cs',
    'science': 'https://www.science.org/rss/news_current.xml',
    'nature': 'https://www.nature.com/subjects/computer-science.rss',
    'sciencemag': 'https://www.sciencemag.org/rss/news_current.xml',
    'newscientist': 'https://www.newscientist.com/subject/technology/feed/',
    'sciencedaily': 'https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml',
    'techcrunch': 'https://techcrunch.com/feed/',
    'thenextweb': 'https://thenextweb.com/feed/',
    'wired': 'https://www.wired.com/feed/rss',
    'venturebeat': 'https://venturebeat.com/feed/',
    'androidauthority': 'https://www.androidauthority.com/feed/',
    'androidcentral': 'https://www.androidcentral.com/rss.xml',
    'androidpolice': 'https://www.androidpolice.com/feed/',
    '9to5 google': 'https://9to5google.com/feed/',
    'gizmodo': 'https://gizmodo.com/rss',
    'theverge': 'https://www.theverge.com/rss/index.xml',
    'engadget': 'https://www.engadget.com/rss.xml',
    'zdnet': 'https://www.zdnet.com/topic/artificial-intelligence/rss.xml',
    'slashdot': 'http://rss.slashdot.org/Slashdot/slashdotMain',
    'techradar': 'https://www.techradar.com/feeds/articletype/news',
    'cnet': 'https://www.cnet.com/rss/news/',
    'xda': 'https://www.xda-developers.com/feed/',
    'arstechnica': 'http://feeds.arstechnica.com/arstechnica/index',
    'readwrite': 'https://readwrite.com/feed/',
    'techrepublic': 'https://www.techrepublic.com/rssfeeds/articles/',
    'motherboard': 'https://www.vice.com/en_us/rss/topic/tech',
    'theatlantic': 'https://www.theatlantic.com/feed/channel/technology/',
    'reuters': 'https://www.reutersagency.com/feed/?best-topics=tech&post_type=best',
    'techtarget': 'https://searchsecurity.techtarget.com/rss/Security-Wire-Daily-News.xml',
    'baseball prospectus': 'https://www.baseballprospectus.com/feed/',
    'baseball america': 'https://www.baseballamerica.com/feed/',
    'mlb trade rumors': 'https://www.mlbtraderumors.com/feed',
    'the athletic': 'https://theathletic.com/feeds/rss/news/',
    'mit technology': 'https://www.technologyreview.com/feed/',
    'techmeme': 'https://www.techmeme.com/feed.xml',
    'darkreading': 'https://www.darkreading.com/rss.xml',
    'electronic frontier foundation': 'https://www.eff.org/rss/updates.xml',
    'ieee spectrum': 'https://spectrum.ieee.org/feeds/feed.rss',
    'gigaom': 'https://gigaom.com/feed/',
    'bleeping computer': 'https://www.bleepingcomputer.com/feed/',
    'theregister': 'https://www.theregister.com/headlines.rss',
    'theguardian': 'https://www.theguardian.com/world/rss',
    'axios': 'https://api.axios.com/feed/',
    'washington post': 'https://feeds.washingtonpost.com/rss/national',
    'the independent': 'https://www.independent.co.uk/news/uk/rss',
    'npr news': 'https://feeds.npr.org/1001/rss.xml',
    'bloomberg markets': 'https://feeds.bloomberg.com/markets/news.rss',
    'the gradient': 'https://thegradient.pub/rss/',
    'kdnuggets': 'https://www.kdnuggets.com/feed',
    'data science central': 'https://www.datasciencecentral.com/category/business-topics/feed/',
    'datafloq': 'https://datafloq.com/feed/',
}

MAX_ARTICLES_PER_FEED = 25  # Limit articles per feed
PROCESSING_TIMEOUT = 290   # Timeout in seconds (adjust as needed)

# Timezone and date parsing utilities
TIMEZONE_OFFSETS = {
    'EST': '-0500',
    'EDT': '-0400',
    'CST': '-0600',
    'CDT': '-0500',
    'MST': '-0700',
    'MDT': '-0600',
    'PST': '-0800',
    'PDT': '-0700',
    'GMT': '+0000',
    'UTC': '+0000'
}

# Regular expression to match ISO 8601 format strings
ISO_8601_REGEX = re.compile(
    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?'
)

def is_article_from_recent_days(article_date):
    """
    Checks if the article's published date is from today, yesterday, or tomorrow (in UTC).

    Args:
    article_date: The published date string of the article.

    Returns:
    True if the article was published on today, yesterday, or tomorrow (in UTC), False otherwise.
    """
    try:
        article_datetime = None

        # Handle ISO 8601 format using regex
        if ISO_8601_REGEX.match(article_date):
            article_datetime = datetime.fromisoformat(article_date)
        else:
            # For RFC 822 and other formats, replace named time zones (e.g., EDT, GMT) with their numeric offsets
            for tz_name, tz_offset in TIMEZONE_OFFSETS.items():
                if tz_name in article_date:
                    article_date = article_date.replace(tz_name, tz_offset)
                    break

            # Parse the date from the article (RFC-822 style)
            article_datetime = datetime.strptime(article_date, '%a, %d %b %Y %H:%M:%S %z')

        # Normalize the article date to UTC
        article_datetime_utc = article_datetime.astimezone(timezone.utc)

        # Get today's date in UTC (timezone-aware, without time)
        today_utc = datetime.now(tz=timezone.utc).date()
        yesterday_utc = today_utc - timedelta(days=1)
        tomorrow_utc = today_utc + timedelta(days=1)

        # Check if the article's UTC date is from today, yesterday, or tomorrow
        return article_datetime_utc.date() in {yesterday_utc, today_utc, tomorrow_utc}

    except Exception as e:
        print(f"Error parsing date: {e}")
        return False

def euclidean_distance(vector1, vector2):
    """
    Calculates the Euclidean distance between two vectors.

    Args:
    vector1: The first vector.
    vector2: The second vector.

    Returns:
    The Euclidean distance between the two vectors.
    """
    # Calculate the squared differences between the corresponding elements of the two vectors.
    squared_differences = [(vector1[i] - vector2[i]) ** 2 for i in range(len(vector1))]

    # Sum the squared differences.
    sum_of_squared_differences = sum(squared_differences)

    # Take the square root of the sum of the squared differences.
    return math.sqrt(sum_of_squared_differences)

async def fetch_feed(session, feed_name, feed_url):
    try:
        async with session.get(feed_url) as response:
            feed_data = await response.text()
            parsed_feed = feedparser.parse(feed_data)
            return feed_name, parsed_feed
    except Exception as e:
        print(f'Error fetching feed {feed_name}: {e}')
        return feed_name, None

async def get_responses():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_feed(session, name, url) for name, url in feeds.items()]
        return await asyncio.gather(*tasks)

async def process_article(article, feed_title):
    try:
        if not is_article_from_recent_days(article.get('published', '')):
            return None

        temp_dict = {}
        temp_dict['feed_title'] = feed_title

        # Process title
        soup = BeautifulSoup(article.get('title', ''), 'html.parser')
        temp_dict['entry_title'] = ''.join(soup.findAll(text=True))
        temp_dict['entry_link'] = article.get('link', '')
        temp_dict['entry_published'] = article.get('published', '')

        # Process summary
        soup = BeautifulSoup(article.get('summary', ''), 'html.parser')
        temp_dict['entry_summary'] = ''.join(soup.findAll(text=True))
        temp_dict['training_data'] = temp_dict['entry_title'] + ' ' + temp_dict['entry_summary']

        # Process tags
        temp_dict['entry_tags'] = article.get('tags', [{'term': 'No tags available'}])
        unique_tags = set()
        filtered_tags = []

        for tag in temp_dict['entry_tags']:
            term_lower = re.sub(r'\s+', ' ', tag['term'].strip().lower())
            if term_lower.startswith('/') or term_lower.startswith('\\'):
                continue
            if term_lower not in unique_tags:
                unique_tags.add(term_lower)
                filtered_tags.append({'term': term_lower})

        temp_dict['entry_tags'] = filtered_tags

        # Process image
        info = None
        media_lookups = ['media_thumbnail', 'media_content']
        for lookup in media_lookups:
            if lookup in article.keys():
                try:
                    news_source_img = article[lookup][0]['url']
                    response = requests.get(news_source_img)
                    image = Image.open(BytesIO(response.content))
                    width, height = image.size
                    info = {
                        'width': width,
                        'height': height,
                        'url': news_source_img
                    }
                    temp_dict['entry_image'] = info
                except:
                    continue
                break
        else:
            try:
                news_source_img = article.get('image', {}).get('href', '')
                if news_source_img:
                    response = requests.get(news_source_img)
                    image = Image.open(BytesIO(response.content))
                    width, height = image.size
                    info = {
                        'width': width,
                        'height': height,
                        'url': news_source_img
                    }
                else:
                    raise ValueError("No image URL found")
                temp_dict['entry_image'] = info
            except:
                info = {
                    'width': 250,
                    'height': 250,
                    'url': 'https://courtneyperigo.com/assets/brittany.png'
                }
                temp_dict['entry_image'] = info

        # Encode training data
        temp_dict['article_vector'] = model.encode(temp_dict['training_data']).tolist()

        return temp_dict
    except Exception as e:
        print(f"Error processing article: {e}")
        return None

async def process_articles(news_dict):
    tasks = []
    for news_source, feed_data in news_dict.items():
        if not feed_data:
            continue
        feed_title = feed_data['feed'].get('title', news_source)
        entries = feed_data['entries']
        
        # Filter entries to only those from recent days
        recent_entries = [article for article in entries if is_article_from_recent_days(article.get('published', ''))]
        
        # Now limit to MAX_ARTICLES_PER_FEED
        limited_entries = recent_entries[:MAX_ARTICLES_PER_FEED]
        
        for article in limited_entries:
            tasks.append(process_article(article, feed_title))
    # Process articles with a timeout
    return await asyncio.gather(*tasks)

@app.route("/fetch-news")
def fetch_news():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Fetch feeds asynchronously
    feed_results = loop.run_until_complete(get_responses())
    news_dict = {name: data for name, data in feed_results if data is not None}

    # Process articles asynchronously with timeout
    try:
        articles = loop.run_until_complete(
            asyncio.wait_for(process_articles(news_dict), timeout=PROCESSING_TIMEOUT)
        )
    except asyncio.TimeoutError:
        print("Processing articles timed out")
        articles = []  # Proceed with whatever has been processed

    # Filter out None results
    retrieved_news = [article for article in articles if article is not None]

    # If no articles were processed, return an error message
    if not retrieved_news:
        return "<h1>No news articles were fetched.</h1>"

    # Proceed with distance calculations
    seed_vectors = [
        model.encode('Tech firms invest in data science to improve operations, customer experience, and marketing'),
        model.encode('The AI Era Accelerates Marketing Agencies From Services To Solutions In 2024'),
        model.encode('Analytics applications for marketing and advertising services'),
        model.encode('Data engineering and technology stack for business intelligence'),
        model.encode('Articles about marketers using machine learning to improve the customer experience')
    ]

    # Calculate distances
    for article in retrieved_news:
        article_distance = min(
            euclidean_distance(seed, article['article_vector']) for seed in seed_vectors
        )
        article['article_min_distance'] = article_distance

    # Sort and store the news
    sorted_news = sorted(retrieved_news, key=lambda k: k['article_min_distance'])

    # Store news in Google Cloud Storage bucket
    gcs = storage.Client()
    bucket = gcs.get_bucket("personal-website-35-machinanova-news")
    gcs_file_string = 'retrieved_news/news.json'
    blob = bucket.blob(gcs_file_string)
    blob.upload_from_string(
        data=json.dumps(sorted_news, indent=4),
        content_type='application/json',
        timeout=300
    )

    return "<h1>News Fetched!</h1>"

# get custom news recommendation based on user input
@app.route("/get-custom-news/v1", methods=['GET'])
def get_custom_news():
    args = request.args
    # get user input
    user_input = args.get('query', '')
    # clean query to remove security risks
    user_input = re.sub(r'[^\x00-\x7f]', r'', user_input)
    # encode user input
    user_vector = model.encode(user_input)
    # get news from GCS bucket
    gcs = storage.Client()
    bucket = gcs.get_bucket("personal-website-35-machinanova-news")
    gcs_file_string = 'retrieved_news/news.json'
    blob = bucket.blob(gcs_file_string)
    # read into dict
    retrieved_news = json.loads(blob.download_as_string())

    # get euclidean distance between user input and each article
    for article in retrieved_news:
        try:
            article['article_min_distance'] = euclidean_distance(user_vector, article['article_vector'])
        except:
            article['article_min_distance'] = float('inf')

    # sort retrieved news by distance
    sorted_news = sorted(retrieved_news, key=lambda k: k['article_min_distance'])

    # return sorted news as json object
    json_news = json.dumps(sorted_news, indent=4, sort_keys=False, allow_nan=False)
    response = make_response(json_news)
    response.headers.set('Content-Type', 'application/json')
    response.headers.set('Access-Control-Allow-Origin', '*')

    return response

# get default news recommendation
@app.route("/get-news/v1", methods=['GET'])
def get_default_news():
    # get news from GCS bucket
    gcs = storage.Client()
    bucket = gcs.get_bucket("personal-website-35-machinanova-news")
    gcs_file_string = 'retrieved_news/news.json'
    blob = bucket.blob(gcs_file_string)
    # read into dict
    retrieved_news = json.loads(blob.download_as_string())

    # return sorted news as json object
    json_news = json.dumps(retrieved_news, indent=4, sort_keys=False, allow_nan=False)
    response = make_response(json_news)
    response.headers.set('Content-Type', 'application/json')
    response.headers.set('Access-Control-Allow-Origin', '*')

    return response

@app.route("/")
def home():
    # return basic html page "hello world"
    return "<h1>News Retriever Works!</h1>"

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)

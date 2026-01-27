from flask import Flask, request, make_response
from google.cloud import storage
import json
import re
import openai
import os

from google.cloud import secretmanager

def _resolve_openai_key(raw: str) -> str:
    """Turn 'projects/.../versions/latest' into the real key, or return raw."""
    if raw.startswith("projects/"):
        sm = secretmanager.SecretManagerServiceClient()
        key_bytes = sm.access_secret_version(request={"name": raw}).payload.data
        return key_bytes.decode("utf-8")
    return raw

openai.api_key = _resolve_openai_key(os.environ["OPENAI_API_KEY"])

app = Flask(__name__)

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
    'HubSpot Marketing Blog': 'https://blog.hubspot.com/marketing/rss.xml',
    'Moz Blog': 'https://moz.com/blog/feed.xml',
    'Semrush Blog': 'https://www.semrush.com/blog/feed/',
    'Chief Martec': 'https://chiefmartec.com/feed',
    'MarTech (News & Insights)': 'https://martech.org/feed/',
    # AI/ML Focused
    'OpenAI Blog': 'https://openai.com/blog/rss.xml',
    'Google AI Blog': 'https://blog.google/technology/ai/rss/',
    'Anthropic News': 'https://www.anthropic.com/rss.xml',
    'Hugging Face Blog': 'https://huggingface.co/blog/feed.xml',
    'DeepMind Blog': 'https://deepmind.google/blog/rss.xml',
    # Marketing + Data Intersection
    'Marketing AI Institute': 'https://www.marketingaiinstitute.com/blog/rss.xml',
    'Econsultancy': 'https://econsultancy.com/feed/',
    'Think with Google': 'https://www.thinkwithgoogle.com/rss/',
    'Content Marketing Institute': 'https://contentmarketinginstitute.com/feed/',
    # Data Engineering/Analytics
    'dbt Blog': 'https://www.getdbt.com/blog/rss.xml',
    'Databricks Blog': 'https://www.databricks.com/feed',
    'Snowflake Blog': 'https://www.snowflake.com/feed/',
    'Data Engineering Weekly': 'https://www.dataengineeringweekly.com/feed'
}

MAX_ARTICLES_PER_FEED = 25 # Limit articles per feed
PROCESSING_TIMEOUT = 290   # Timeout in seconds (adjust as needed)

# Seed topics for ranking articles by relevance
SEED_TOPICS = [
    'Tech firms invest in data science to improve operations, customer experience, and marketing',
    'The AI Era Accelerates Marketing Agencies From Services To Solutions In 2024',
    'Analytics applications for marketing and advertising services',
    'Data engineering and technology stack for business intelligence',
    'Articles about marketers using machine learning to improve the customer experience'
]

# Cache for seed vectors (computed once at startup)
_seed_vectors_cache = None

def get_seed_vectors():
    """Get seed vectors, computing them once and caching for reuse."""
    global _seed_vectors_cache
    if _seed_vectors_cache is None:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=SEED_TOPICS  # batch all seeds in one API call
        )
        _seed_vectors_cache = [item.embedding for item in response.data]
    return _seed_vectors_cache

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
    from datetime import datetime, timezone, timedelta

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

def cosine_similarity(vec1, vec2):
    """
    Calculates cosine similarity between two vectors.
    OpenAI embeddings are normalized, so dot product equals cosine similarity.

    Returns a value between -1 and 1, where 1 means identical.
    """
    return sum(a * b for a, b in zip(vec1, vec2))

@app.route("/fetch-news")
def fetch_news():
    # Only allow Cloud Scheduler to trigger this in production
    if os.environ.get('GAE_ENV', '') == 'standard':
        if request.headers.get('X-Appengine-Cron') != 'true':
            return "Forbidden", 403

    import asyncio
    import aiohttp
    import feedparser
    from bs4 import BeautifulSoup

    

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

            # Process image - just store URL without fetching (avoids blocking sync calls)
            default_image = 'https://courtneyperigo.com/assets/brittany.png'
            image_url = default_image
            for lookup in ['media_thumbnail', 'media_content']:
                if lookup in article and article[lookup]:
                    try:
                        image_url = article[lookup][0]['url']
                        break
                    except (KeyError, IndexError):
                        continue
            temp_dict['entry_image'] = {'url': image_url}

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

    # Batch embed all articles in one API call (major cost savings)
    training_texts = [article['training_data'] for article in retrieved_news]

    # OpenAI supports up to 2048 inputs per call, batch in chunks if needed
    BATCH_SIZE = 2000
    all_embeddings = []
    for i in range(0, len(training_texts), BATCH_SIZE):
        batch = training_texts[i:i + BATCH_SIZE]
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])

    # Assign embeddings to articles
    for article, embedding in zip(retrieved_news, all_embeddings):
        article['article_vector'] = embedding

    # Get cached seed vectors (computed once, reused across requests)
    seed_vectors = get_seed_vectors()

    # Calculate similarity scores (higher is better)
    for article in retrieved_news:
        article_similarity = max(
            cosine_similarity(seed, article['article_vector']) for seed in seed_vectors
        )
        article['article_max_similarity'] = article_similarity

    # Sort by highest similarity first
    sorted_news = sorted(retrieved_news, key=lambda k: k['article_max_similarity'], reverse=True)

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

@app.route("/get-news/v1", methods=['GET'])
def get_default_news():
    from google.cloud import storage

    # Get news from GCS bucket
    gcs = storage.Client()
    bucket = gcs.get_bucket("personal-website-35-machinanova-news")
    gcs_file_string = 'retrieved_news/news.json'
    blob = bucket.blob(gcs_file_string)
    retrieved_news = json.loads(blob.download_as_string())

    json_news = json.dumps(retrieved_news, indent=4, sort_keys=False, allow_nan=False)
    response = make_response(json_news)
    response.headers.set('Content-Type', 'application/json')
    response.headers.set('Access-Control-Allow-Origin', '*')

    return response

@app.route("/")
def home():
    return "<h1>News Retriever Works!</h1>"

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)

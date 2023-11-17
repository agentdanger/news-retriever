from flask import Flask, request, make_response
from google.cloud import storage
import feedparser
from PIL import Image
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import math
import numpy as np
import json
import re

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')

# get euclidean distance between two vectors

def euclidean_distance(vector1, vector2):
    """
    Calculates the euclidean distance between two vectors.

    Args:
    vector1: The first vector.
    vector2: The second vector.

    Returns:
    The euclidean distance between the two vectors.
    """

    # Calculate the squared differences between the corresponding elements of the two vectors.
    squared_differences = [
    (vector1[i] - vector2[i])**2 for i in range(len(vector1))
    ]

    # Sum the squared differences.
    sum_of_squared_differences = sum(squared_differences)

    # Take the square root of the sum of the squared differences.
    euclidean_distance = math.sqrt(sum_of_squared_differences)

    return euclidean_distance

# define feeds

feeds = {
    'analytics_vidhya': 'https://www.analyticsvidhya.com/feed/',
    'dataquest': 'https://www.dataquest.io/blog/feed/',
    'dataconomy': 'https://dataconomy.com/feed/',
    'machinelearningmastery': 'https://machinelearningmastery.com/feed/',
    'medium': 'https://medium.com/feed/tag/data-science',
    'towardsdatascience': 'https://towardsdatascience.com/feed',
    'adexchanger': 'https://adexchanger.com/feed/',
    'adweek': 'https://www.adweek.com/feed/',
    'wsj': 'https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml',
    'nyt': 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'cnn': 'http://rss.cnn.com/rss/cnn_topstories.rss',
    'foxnews': 'http://feeds.foxnews.com/foxnews/latest',
    'nbcnews': 'http://feeds.nbcnews.com/nbcnews/public/news',
    'bbc': 'http://feeds.bbci.co.uk/news/rss.xml',
    'apnews_topnews': 'https://rsshub.app/apnews/topics/ap-top-news',
    'apnews_tech': 'https://rsshub.app/apnews/topics/technology',
    'bloomberg': 'https://feeds.bloomberg.com/technology/news.rss',
    'businessinsider': 'https://markets.businessinsider.com/rss/news',
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
    '9to5google': 'https://9to5google.com/feed/',
    'techcrunch': 'https://techcrunch.com/feed/',
    'gizmodo': 'https://gizmodo.com/rss',
    'theverge': 'https://www.theverge.com/rss/index.xml',
    'engadget': 'https://www.engadget.com/rss.xml',
    'zdnet': 'https://www.zdnet.com/topic/artificial-intelligence/rss.xml',
    'techmeme': 'https://www.techmeme.com/feed.xml',
    'slashdot': 'http://rss.slashdot.org/Slashdot/slashdotMain',
}

# get response from all feeds in json format
def get_response():
    response = {}
    for feed in feeds:
        response[feed] = feedparser.parse(feeds[feed])
    return response

@app.route("/fetch-news")
def fetch_news():
    # Fetch news from news API and encode it using sentence-transformers
    news_dict = get_response()

    retrieved_news = []

    for news_source in feeds.keys():
        # get feed image
        info = None
        try:
            feed_title = news_dict[news_source]['feed']['title']
            # remove ascii characters from feed title
            feed_title = re.sub(r'[^\x00-\x7f]',r'', feed_title)
        except:
            print('no feed title: ' + news_source)
            continue
        for article in news_dict[news_source]['entries']:
            temp_dict = {}
            try:
                temp_dict['feed_title'] = feed_title
                # clean article title remove html tags using beautiful soup
                soup = BeautifulSoup(article['title'], 'html.parser')
                texts = soup.findAll(text=True)
                temp_dict['entry_title'] = ''.join(texts)
                # remove ascii characters from article title
                temp_dict['entry_title'] = re.sub(r'[^\x00-\x7f]',r'', temp_dict['entry_title'])
                temp_dict['entry_link'] = article['link']
                temp_dict['entry_published'] = article['published']
            except:
                continue
            try:
                # clean article summary remove html tags using beautiful soup
                soup = BeautifulSoup(article['summary'], 'html.parser')
                texts = soup.findAll(text=True)
                temp_dict['entry_summary'] = ''.join(texts)
                # remove ascii characters from article summary
                temp_dict['entry_summary'] = re.sub(r'[^\x00-\x7f]',r'', temp_dict['entry_summary'])
                # remove "\n" and "[]" from article summary
                temp_dict['entry_summary'] = temp_dict['entry_summary'].replace('\n', '')
                temp_dict['entry_summary'] = temp_dict['entry_summary'].replace('[]', '')
                temp_dict['training_data'] = temp_dict['entry_title'] + ' ' + temp_dict['entry_summary']
            except:
                temp_dict['entry_summary'] = 'No summary available'
            try:
                temp_dict['entry_tags'] = article['tags']
            except:
                temp_dict['entry_tags'] = [
                    {
                        'term': 'No tags available'
                    }
                ]
            # get article image
            info = None
            media_lookups = [
                'media_thumbnail',
                'media_content',
            ]
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
                    continue
            else:
                try:
                    news_source_img = news_dict[news_source]['feed']['image']['href']
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
                    info = {
                        'width': 250,
                        'height': 250,
                        'url': 'https://courtneyperigo.com/assets/brittany.jpg'
                    }
                    temp_dict['entry_image'] = info
                    continue
            # encode training data
            try:
                temp_dict['article_vector'] = model.encode(temp_dict['training_data'])
            except:
                temp_dict['article_vector'] = [99.0 for i in range(384)]

            retrieved_news.append(temp_dict)
    
    seed_vectors = []

    # hardcode data science seed vectors (for now)
    seed_vectors.append(model.encode('Tech firms invest in data science to imporove operations, customer experience, and marketing'))
    seed_vectors.append(model.encode('The AI Era Accelerates Agencies From Services To Solutions In 2024'))
    seed_vectors.append(model.encode('Analytics applications for marketing and advertising services'))
    seed_vectors.append(model.encode("Articles about marketers using machine learning to improve the customer experience"))

    for article in retrieved_news:
        article_distance = 999999999999
        distances = []
        for seed in seed_vectors:
            try:
                temp_euc_distance = euclidean_distance(seed, article['article_vector'])
                distances.append(temp_euc_distance)
            except:
                continue
            if temp_euc_distance < article_distance:
                article_distance = temp_euc_distance
        article['article_min_distance'] = article_distance
        article['article_distance'] = np.average(distances)
        try:
            article['article_vector'] = article['article_vector'].tolist()
        except:
            article['article_vector'] = [99.0 for i in range(384)]

    # sort retrieved news by distance
    sorted_news = sorted(retrieved_news, key=lambda k: k['article_distance'])

    # Store news in Google Cloud Storage bucket
    gcs = storage.Client()
    bucket = gcs.get_bucket("personal-website-35-machinanova-news")
    gcs_file_string = 'retrieved_news/news.json'
    blob = bucket.blob(gcs_file_string)
    blob.upload_from_string(data=json.dumps(sorted_news, 
                                            indent=4
                                            ), 
                                            content_type='application/json',
                                            timeout=300)

    # return basic html page "news fetched"
    return "<h1>News Fetched!</h1>"

# get custom news recommendation based on user input
@app.route("/get-custom-news/v1", methods=['GET'])
def get_custom_news():
    args = request.args
    # get user input
    user_input = args['query']
    # clean query to remove security risks
    user_input = re.sub(r'[^\x00-\x7f]',r'', user_input)
    print(user_input)
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
            article['user_distance'] = 999999999999
    
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
    
    # sort retrieved news by distance
    sorted_news = sorted(retrieved_news, key=lambda k: k['article_min_distance'])

    # return sorted news as json object
    json_news = json.dumps(sorted_news, indent=4, sort_keys=False, allow_nan=False)
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
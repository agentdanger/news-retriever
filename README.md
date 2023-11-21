# MachinaNova3: News Fetcher Application
This is a simple application that fetches news articles from a news API and stores them in a database. The application is deployed on Google App Engine and runs every 30 minutes. The application is written in Python and uses Flask as the web framework. News files are stored in Google Cloud Storage for retrieval by CourtneyPerigo.com's news page.

## About the News Fetcher Machine Learning Model
The News Fetcher Machine Learning Model leverages a pre-built sentence transformer model to create embeddings for news articles. The embeddings are then used to find similar articles using euclidean distance. More information about the underlying model can be found on the [Hugging Face Model Hub](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

# MachinaNova3: News Fetcher Application
This is a simple application that fetches news articles from a news API and stores them in a database. The application is deployed on Google App Engine and runs every 30 minutes. The application is written in Python and uses Flask as the web framework. News files are stored in Google Cloud Storage for retrieval by CourtneyPerigo.com's news page.

## Requirements
see requriements.txt

## Create Virtual Environment
Note that this will create a virtual environment with the directory .\env
Windows instructions below.
```
python -m venv env
.\env\Scripts\activate
```

## Install requirements
```
pip install -r requirements.txt
```

## Run the app locally
```
flask --app main run
```

## Deploying the App to Google App Engine
```
gcloud app deploy
```

## Deploying the cron.yaml to Google App Engine
```
gcloud app deploy cron.yaml
```
from newspaper import Article
from newsapi import NewsApiClient  # API key: 6321747c754345d684ff295c8c93cea6
import nltk
import sys

nltk.download('punkt')


class ArticleData:

    def __init__(self, url):
        article = Article(url)
        article.download()
        article.parse()

        self.date = article.publish_date
        self.text = article.text

        article.nlp()
        self.keywords = article.keywords
        self.summary = article.summary

    def get_date(self):
        return self.date

    def get_text(self):
        return self.text

    def get_keywords(self):
        return self.keywords

    def get_summary(self):
        return self.summary


class ArticleScraper:

    def __init__(self, key):
        self.client = NewsApiClient(key)

    # find article urls from keyword array
    def get_articles(self, keywords):

        # make keyword list into "keyword1 AND keyword2 AND ..." for newsapi functionality
        res = "".join(keywords[0])
        for keyword in keywords[1:]:
            add = " AND " + keyword
            res = res + add
      
        articles = self.client.get_everything(q=res, language="en", sort_by="relevancy")

        url_list = []
        for article in articles["articles"]:
            url_list.append(article["url"])

        return url_list

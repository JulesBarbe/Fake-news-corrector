from newspaper import Article
from newsapi import NewsApiClient
#API key: 6321747c754345d684ff295c8c93cea6

class Article():

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


class ArticleScraper():

    def __init__(self, key):
        self.client = NewsApiClient(key)

    


 
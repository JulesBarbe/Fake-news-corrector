from newspaper import Article
from newsapi import NewsApiClient   #API key: 6321747c754345d684ff295c8c93cea6
import nltk
nltk.download('punkt')


class Article_Data():

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

    # find articles from keyword array
    def get_articles(self, keywords):

            return 0



if __name__ == "__main__":
    newsapi = NewsApiClient(api_key='6321747c754345d684ff295c8c93cea6')
    top_headlines = newsapi.get_top_headlines(
    q="World War",
    language='en',
)

    print(top_headlines["articles"][0]["url"])

    #https://www.aljazeera.com/economy/2021/4/7/vaccine-policy-is-economic-policy-imf-chief-stresses

    article = Article_Data("https://www.aljazeera.com/economy/2021/4/7/vaccine-policy-is-economic-policy-imf-chief-stresses")
    print(article.get_date())
    print(article.get_keywords())
 
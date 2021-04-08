from flask import Flask, render_template, request
import articles
import predict
import pickle
 #API key: 6321747c754345d684ff295c8c93cea6 for newsapi

app = Flask(__name__)

with open('Models/tfidf_vectorizer', 'rb') as file:
    vectorizer = pickle.load(file)

with open('Models/SGD_model', 'rb') as file:
    sgd = pickle.load(file)

with open('Models/LSA_model', 'rb') as file:
    lsa = pickle.load(file)

fn = predict.Fake_news(vectorizer, sgd, lsa)
as = articles.ArticleScraper("6321747c754345d684ff295c8c93cea6")

@app.route('/', methods=['GET'])
def load():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def parse():
    url = request.form['url']
    article = articles.Article_Data(url)
    text = article.get_text()
    ptext = fn.preprocess([text])
    label = fn.classify(ptext)

    # not fake news
    if label == 1:

        return "\n\n".join(["Article seems reliable.", "Article text:", "=" * 86, text])

    # fake news
    else:

        keywords = article.get_keywords()
        alt_article_urls = as.get_articles(keywords)
        curr_topics = fn.get_topics(ptext)
        
        least = 2^20    # big value

        for article_url in alt_article_urls:
            alt_article = articles.Article_Data(article_url)
            alt_topics = fn.get_topics(fn.preprocess(alt_article.get_text()))
            dist = fn.topic_distance(curr_topics, alt_topics)
            if dist < least:
                alt = article_url, alt_article.get_summary()        # idk how to work with this alt here in python since we dont need to declare it in advance
        


        return "\n\n".join(["Article seems unreliable.", "Alternate article url:", "=" * 86, alt])


if __name__ == "__main__":
    app.run()

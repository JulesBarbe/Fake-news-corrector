from flask import Flask, render_template, request
import articles
import predict
import pickle

# API key: 6321747c754345d684ff295c8c93cea6 for newsapi

# Launch Flask app
app = Flask(__name__)

# Load models
with open('Models/tfidf_vectorizer', 'rb') as file:
    vectorizer = pickle.load(file)

with open('Models/SGD_model', 'rb') as file:
    sgd = pickle.load(file)

with open('Models/LSA_model', 'rb') as file:
    lsa = pickle.load(file)

# Create FakeNews and ArticleScraper objects
fn = predict.FakeNews(vectorizer, sgd, lsa)
scraper = articles.ArticleScraper("6321747c754345d684ff295c8c93cea6")


# GET and POST for normal Flask operation: user inputs url through textbox
@app.route('/', methods=['GET'])
def load():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def parse():
    url = request.form['url']
    return predict(url)


# GET and POST for external operation through chrome extension: predicts for active page
@app.route('/external', methods=['GET'])
def load_external():
    url = request.args.get("url")
    return predict(url)


@app.route('/external', methods=['POST'])
def get_request():
    return render_template("index.html")


# Prediction function
def predict(url):
    article = articles.ArticleData(url)
    text = article.get_text()
    ptext = fn.preprocess([text])
    label = fn.classify(ptext)

    # not fake news
    if label == 1:
        return "\n\n".join(["Article seems reliable.", "Article text:", "=" * 86, text])

    # fake news
    else:
        keywords = article.get_keywords()
        alt_article_urls = scraper.get_articles(keywords)
        curr_topics = fn.get_topics(ptext)

        least = float('inf')  # big value

        alt = "No article found"
        # print("Urls: ", alt_article_urls)

        for article_url in alt_article_urls:
            alt_article = articles.ArticleData(article_url)
            alt_topics = fn.get_topics(fn.preprocess([alt_article.get_text()]))
            dist = fn.topic_distance(curr_topics, alt_topics)
            if article_url == url:
                continue
            if dist < least:
                # idk how to work with this alt here in python since we dont need to declare it in advance
                alt = article_url

        return "<br/>".join(["Article seems unreliable.</br>", "Here is a more reliable source on the same topic:", alt,
                             "", "Article text: <br/>", "=" * 86, "", alt_article.get_summary()])


if __name__ == "__main__":
    app.run(debug=True)

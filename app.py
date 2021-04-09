from flask import Flask, render_template, request
import articles
import predict
import pickle
import sys

api_key = "a3fb78e592df4927b3014f97d193cd42" # this is Zahur's
# API key: 6321747c754345d684ff295c8c93cea6 for newsapi
graham = "The FitnessGram Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as " \
         "it continues. The 20 meter pacer test will begin in 30 seconds. Line up at the start. The running speed " \
         "starts slowly but gets faster each minute after you hear this signal bodeboop. A sing lap should be " \
         "completed every time you hear this sound. ding Remember to run in a straight line and run as long as " \
         "possible. The second time you fail to complete a lap before the sound, your test is over. The test will " \
         "begin on the word start. On your mark. Get ready!… Start. "

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
scraper = articles.ArticleScraper(api_key)


# GET and POST for normal Flask operation: user inputs url through textbox
@app.route('/', methods=['GET'])
def load():
    return render_template('home.html')


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
    return render_template("home.html")


# Prediction function
def predict(url):
    article = articles.ArticleData(url)
    text = article.get_text()
    ptext = fn.preprocess([text])
    label = fn.classify(ptext)

    # not fake news
    if label == 1:
        return render_template("true.html", text=text)

    # fake news
    else:
        # get keyword list from current article
        keywords = article.get_keywords()[:5]
        print("Number of keywords: " + str(len(keywords)), file=sys.stderr)

        # get alternate articles using keywords
        alt_article_urls = scraper.get_articles(keywords)
        print("Number of articles found: " + str(len(alt_article_urls)), file=sys.stderr)

        # get LSA topics of current article to compare to alternate choices
        curr_topics = fn.get_topics(ptext)

        # Loop through alternate articles and return the one with least topic distance
        least = float('inf')  # big value
        alt = 0
        res_article = article
        n = 0
        alt_article = None

        for article_url in alt_article_urls:

            n += 1
            print("Article " + str(n), file=sys.stderr)

            # in case alternate article can't be downloaded
            try:
                alt_article = articles.ArticleData(article_url)
            except:
                print("Skipped downloading alternate article", file=sys.stderr)
                continue

            # get alternate article text and calculate distance
            alt_text = fn.preprocess([alt_article.get_text()])
            alt_topics = fn.get_topics(alt_text)
            dist = fn.topic_distance(curr_topics, alt_topics)

            # don't return same article
            if article_url == url:
                print("Same url", file=sys.stderr)
                continue

            # only return "unfake" articles (unecessary with the quality of our classifier :)))) )
            if fn.classify(alt_text) == -1:
                print("Skipped alternate article", file=sys.stderr)
                continue

            # replace if minimum distance
            if dist < least:
                least = dist
                alt = article_url
                res_article = alt_article
                print("Found one!", file=sys.stderr)

        # if no alternate article found
        if alt == 0:
            alt_text = "No alternate article was found. Enjoy this instead: " + graham
            return render_template("fake.html", text=text, alt_url="<No url found.>", alt_text=alt_text)
        else:
            return render_template("fake.html", text=text, alt_url=alt, alt_text=res_article.get_summary())


if __name__ == "__main__":
    app.run(debug=True)

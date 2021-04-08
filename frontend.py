from flask import Flask, render_template, request
import articles
import predict
import pickle

app = Flask(__name__)

with open('Models/tfidf_vectorizer', 'rb') as file:
    vectorizer = pickle.load(file)

with open('Models/SGD_model', 'rb') as file:
    sgd = pickle.load(file)

with open('Models/LSA_model', 'rb') as file:
    lsa = pickle.load(file)

fn = predict.Fake_news(vectorizer, sgd, lsa)


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
        date = article.get_date()

        # find alternate articles

        curr_topics = fn.get_topics(ptext)

        # compare topics among different articles 

        # return url, summary of article with least topic distance from curr_topics
        return "\n\n".join(["Article seems unreliable.", "Article text:", "=" * 86, text])


if __name__ == "__main__":
    app.run()

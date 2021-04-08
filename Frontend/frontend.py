from flask import Flask, render_template, request
from newspaper import Article
app = Flask(__name__)


@app.route('/', methods=['GET'])
def load():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def parse():
    url = request.form['url']
    print(url)
    article = Article(url)
    article.download()
    article.parse()

    text = article.text

    

    return article.text


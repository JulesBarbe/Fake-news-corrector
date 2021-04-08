from flask import Flask, render_template, request
from articles import *
app = Flask(__name__)


@app.route('/', methods=['GET'])
def load():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def parse():
    url = request.form['url']
    data = Article_Data(url)

    return data.get_text()


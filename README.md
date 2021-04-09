# Fake News Corrector
ML project for the McGill Artificial Intelligence Society's machine 
learning bootcamp (MAIS 202, Winter 2021). Training data from Kaggle:
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset.

## Project description
This web app classifies news articles as fake or not given their URL. 
If the article is deemed fake, it will scrape news sites for trustworthy 
articles on a similar topic and return the one with the highest topic similarity 
through Latent Semantic Analysis. Due to limitations of the original 
dataset, results are unreliable - use with caution!

## Running the app 
This web app is available in two flavours: as a pure Flask web app that
allows a user to paste in an article's URL to have its reliability judged,
or as a Google Chrome extension using the Flask app as a backend that allows a user to check whether or not 
the current page is a reliable article.

### Web app:
1. Download the project from GitHub
2. Run app.py, installing necessary dependencies as needed
3. Click the link that appears in the console or navigate to http://localhost:5000
4. Paste the URL of the article you'd like to classify into the text box!

### Chrome extension:
1. Download the project from GitHub
2. In Google Chrome, open chrome://extensions and enable "Developer mode" in the 
   top right corner    
3. Click "Load unpacked" and select the ChromeExtension folder within the downloaded project  files
4. Run app.py, installing necessary dependencies as needed
5. Navigate to the news article you'd like to classify, click on the Fake News 
   Corrector extension in the Extensions bar in the top right, and click the button!
   
### Dependencies:
1. For the newspaper/Article library, run

         pip3 install newspaper3k
      
2. For the News API, run 

         pip install newsapi-python
   
## Repository organization
Below is a listing of relevant files:

      Fake-news-correcter/   
      ├─ ChromeExtension/  
      │  ├─ manifest.json  
      │  ├─ popup.html  
      │  ├─ popup.js  
      ├─ Deliverables/  
      │  ├─ Data Selection Proposal.pdf  
      │  ├─ Deliverable2.pdf  
      │  ├─ Deliverable3.pdf  
      ├─ Models/  
      │  ├─ BenchmarkModels
      │  │  ├─ ...
      │  ├─ ...
      ├─ templates  
      │  ├─ index.html  
      ├─ app.py  
      ├─ articles.py
      ├─ predict.py
      ├─ README.md  



# Fake News Corrector
ML project for the McGill Artificial Intelligence Society's machine 
learning bootcamp (MAIS 202, Winter 2021). Training data from Kaggle:
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset.

![Fake News Detector - home screen with text field for URL](https://github.com/JulesBarbe/Fake-news-corrector/blob/main/Images/fake%20news%20detector2.png)
![Fake News Detector - results for fake article](https://github.com/JulesBarbe/Fake-news-corrector/blob/main/Images/fake%20news%20detector1.png)
![Fake News Detector - Chrome extension visuals](https://github.com/JulesBarbe/Fake-news-corrector/blob/main/Images/fake%20news%20detector4.png)

## Project description
This web app classifies news articles as fake or not given their URL. 
If the article is deemed fake, it will scrape news sites for trustworthy 
articles on a similar topic and return the one with the highest topic similarity 
through Latent Semantic Analysis. Due to limitations of the original 
dataset, results are not always reliable - use with caution!

## Running the app online through Heroku
This web app is available as a pure Flask web app hosted on Heroku that allows a user to paste in an article's URL to have its reliability judged,
and as a Google Chrome extension using the Flask app as a backend that allows a user to check whether or not 
the current page is a reliable article.

### Web app
To run the webapp, simply go to https://fake-news-corrector.herokuapp.com and enter the URL of the article you'd like
to judge the reliability of. No downloads necessary!

### Chrome extension
To run the Chrome Extension:
1. First download the "ChromeExtension" folder from this Github.
2. In Google Chrome, open chrome://extensions and enable "Developer mode" in the 
   top right corner    
3. Click "Load unpacked" and select the ChromeExtension folder.
4. Navigate to the article you'd like to judge the reliability of.
5. Click the Fake News Corrector extension in the Extensions bar in the top right of the Google Chrome window to get your results!

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
      │  ├─ home.html  
      │  ├─ fake.html
      │  ├─ true.html
      ├─ app.py  
      ├─ articles.py
      ├─ predict.py
      ├─ README.md  



from flask import Flask, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import re, os
import warnings
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
# Load different summarization models
from transformers import pipeline


warnings.filterwarnings("ignore")

app = Flask(__name__)
stemmer = PorterStemmer()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tfidf_vectorizer = TfidfVectorizer()

def fetch_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    article_text = ' '.join([paragraph.text for paragraph in paragraphs])
    return article_text


def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocessing():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'data.csv'))
    df1 =df[['article_id','title','description','url']]
    df1['title'] = df1['title'].astype(str)
    df1['description'] = df1['description'].astype(str)
    df1['url'] = df1['url'].astype(str)
    df1['title'] = df1['title'].apply(clean_text)
    df1['description'] = df1['description'].apply(clean_text)
    df1['title'] = df1['title'].str.lower()
    df1['description'] = df1['description'].str.lower()
    
    # Tokenization and remove stop words
    
    df1['title'] = df1['title'].apply(lambda x: [word for word in x.split() if word not in stop_words])
    df1['description'] = df1['description'].apply(lambda x: [word for word in x.split() if word not in stop_words])
    # Stemming
    
    df1['title'] = df1['title'].apply(lambda x: [stemmer.stem(word) for word in x])
    df1['description'] = df1['description'].apply(lambda x: [stemmer.stem(word) for word in x])
    df1['combined_text'] = df1['title'] + df1['description']
    df1['combined_text'] = df1['combined_text'].apply(lambda x: ' '.join(x))
    # TF-IDF Vectorization
    tfidf_matrix = tfidf_vectorizer.fit_transform(df1['combined_text'])
    return df, tfidf_matrix


def query(df, tfidf_matrix, user_query):
    user_query = clean_text(user_query)
    user_query = user_query.lower()
    user_query = [stemmer.stem(word) for word in user_query.split() if word not in stop_words]
    # Vectorize user query
    user_query_vector = tfidf_vectorizer.transform([' '.join(user_query)])
    cosine_similarities = cosine_similarity(user_query_vector, tfidf_matrix).flatten()
    top_n_indices = cosine_similarities.argsort()[-3:][::-1]
    top_articles = df.iloc[top_n_indices]
    best_article_url = top_articles.iloc[0]['url']
    article_text = fetch_article_text(best_article_url)
    return top_articles, article_text


def summary(article_text):
    # Initialize summarization pipelines with different models
    summarizer_model1 = pipeline("summarization", model="facebook/bart-large-cnn")
    # Generate summaries using different models
    summary_model1 = summarizer_model1(article_text, max_length=400, min_length=100, length_penalty=2.0, num_beams=4)
    return summary_model1[0]['summary_text']



@app.route('/')
def index():
    return render_template("index.html")


if (__name__=="__main__"):
    df, tfidf_matrix = preprocessing()
    top_articles, article_text = query(df, tfidf_matrix, "Narendra Modi")
    summary_text = summary(article_text)
    print(top_articles, summary_text)
    
    # app.run(debug=True)
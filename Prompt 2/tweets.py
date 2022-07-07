import streamlit as st
from streamlit import session_state as sn_state
import tweepy
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import urllib.request
import csv

#Title Specification for Streamlit App and Input text box with #minority as default input
st.title("Twitter Sentiment Analysis")
hashtag_search = st.text_input('Enter Hashtag to Search', '#minority')

#This is true when the streamlit app loads first time- instantiating all the session state variables
if 'fetch' not in sn_state or 'result' not in sn_state:
    sn_state['fetch'] = False
    sn_state['result'] = ""
    sn_state['Tweet'] = ""
    sn_state["MultiSelect"] = False

#runs once to instantiate the tokenizer and model for Sentiment Analysis. 
@st.experimental_singleton
def get_model_tokenizer():    
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return model, tokenizer

#Fetch tweets using Tweepy (search_recent_tweets). Limitation of this method - retrieves tweets from last 7 days
#Form a dataframe with user details, tweet info and tweet created datetime
def fetchtweetsandsave(search_string):
    client = tweepy.Client(bearer_token='dd', wait_on_rate_limit=True)
    minority_tweets = []
    paginator = tweepy.Paginator(client.search_recent_tweets, 
                                    query = '#minority -is:retweet lang:en',
                                    user_fields = ['username', 'description', 'location'],
                                    tweet_fields = ['created_at', 'text'],
                                    expansions = 'author_id',
                                max_results=100,limit = 5)

    for page in paginator:
        minority_tweets.append(page)    
    
    result = []
    user_dict = {}

    for response in minority_tweets:    
        for user in response.includes['users']:        
            user_dict[user.id] = {'username': user.username,                               
                                'description': user.description,
                                'location': user.location
                                }
        for tweet in response.data:        
            author_info = user_dict[tweet.author_id]        
            result.append({'UserName': author_info['username'], 
                        'Tweet': tweet.text,
                        'Date': tweet.created_at
                        })


    df = pd.DataFrame(result)
    return df
    #df.to_csv('minoritytweets_0705.csv', encoding='utf-8', index=False)

#Clean the tweets
def preprocess(text):
    new_text = [] 
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

#run the sentiment analysis model and fetch polarity score given a tweet
def SentimentAnalysis(tweet):
    model, tokenizer = get_model_tokenizer()
    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]
    
    text = preprocess(tweet)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    data = {"Labels": labels, "Scores" : scores}
    
    
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    df = pd.DataFrame(data)
    df = df.set_index('Labels')
    st.bar_chart(df)    

#streamlit button click method defn
def gentext_click():    
    sn_state['result'] = fetchtweetsandsave(hashtag_search)
    sn_state['fetch'] = True

#streamlit button specification  
generate = st.button('Fetch Tweets',on_click= gentext_click) 

#loads tweets in a table and also loads a multiselect box to pick an index of the tweet for sentiment analysis
if sn_state['fetch'] == True :
    data = sn_state['result']
    st.write(data)
    selected_indices = st.multiselect('Select one at a time:', data.index)
    sn_state["MultiSelect"] = True        
else:
    st.write("")

#Runs sentiment analysis and loads the scores as a barchart
if sn_state["MultiSelect"] == True:
    data = sn_state['result']
    if(len(selected_indices) > 0):
        sn_state['Tweet'] = data["Tweet"][selected_indices[0]]
        st.info(sn_state['Tweet'])
        SentimentAnalysis(sn_state['Tweet'])
        
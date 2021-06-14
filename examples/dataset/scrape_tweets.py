'''
The following is a basic twitter scraper code using tweepy.
We preprocess the text - 1. Remove Emojis 2. Remove urls from the tweet.
We store the output tweets with tweet-id and tweet-text in each line tab seperated.

You will need to have an active Twitter account and provide your consumer key, secret and a callback url.
You can get your keys from here: https://developer.twitter.com/en/portal/projects-and-apps
Twitter by default implements rate limiting of scraping tweets per hour: https://developer.twitter.com/en/docs/twitter-api/rate-limits
Default limits are 300 calls (in every 15 mins).

Install tweepy (pip install tweepy) to run the code below.
python scrape_tweets.py
'''

import tweepy
import csv
import pickle
import tqdm
import re

#### Twitter Account Details
consumer_key = 'XXXXXXXX' # Your twitter consumer key
consumer_secret = 'XXXXXXXX' # Your twitter consumer secret
callback_url = 'XXXXXXXX' # callback url

#### Input/Output Details
input_file = "input-tweets.tsv" # Tab seperated file containing twitter tweet-id in each line 
output_file = "201509-tweet-scraped-ids-test.txt" # output file which you wish to save

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def de_emojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def preprocessing(text):
    return re.sub(r"http\S+", "", de_emojify(text).replace("\n", "")).strip()

def update_tweet_dict(tweets, tweet_dict):
    for tweet in tweets:
        if tweet:
            try:
                idx = tweet.id_str.strip()
                tweet_dict[idx] = preprocessing(tweet.text)
            except:
                continue
    
    return tweet_dict

def write_dict_to_file(filename, dic):
    with open(filename, "w") as outfile:
        outfile.write("\n".join((idx + "\t" + text) for idx, text in dic.items()))

### Main Code starts here
auth = tweepy.OAuthHandler(consumer_key, consumer_secret, callback_url)
try:
    redirect_url = auth.get_authorization_url()
except tweepy.TweepError:
    print('Error! Failed to get request token.')

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

all_tweets = []
tweets = []
tweet_dict = {}

reader = csv.reader(open(input_file, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
for row in reader:
    all_tweets.append(row[0])

generator = chunks(all_tweets, 100)
batches = int(len(all_tweets)/100)
total = batches if len(all_tweets) % 100 == 0 else batches + 1 

print("Retrieving Tweets...")
for idx, tweet_id_chunks in enumerate(tqdm.tqdm(generator, total=total)):

    if idx >= 300 and idx % 300 == 0: # Rate-limiting every 300 calls (in 15 mins)
        print("Preprocessing Text...")
        tweet_dict = update_tweet_dict(tweets, tweet_dict)
        write_dict_to_file(output_file, tweet_dict)
    
    tweets += api.statuses_lookup(id_=tweet_id_chunks, include_entities=True, trim_user=True, map=None)

print("Preprocessing Text...")
tweet_dict = update_tweet_dict(tweets, tweet_dict)
write_dict_to_file(output_file, tweet_dict)
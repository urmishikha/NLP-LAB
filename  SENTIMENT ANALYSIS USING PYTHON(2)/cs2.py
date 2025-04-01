from detoxify import Detoxify
import pandas as pd
import re

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
tweets = [
    "New phone is awesome, loving the features!",
    "These people are all idiots, they ruin everything.",
    "Women shouldn’t be in tech, they’re useless.",
    "Great game last night, team played well.",
    "This group is a bunch of filthy animals, deport them!",
    "My laptop broke again, so annoying.",
    "Guys in this office are the best, great teamwork.",
    "All these foreigners are stealing our jobs, disgusting."
]

def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r"@\w+|#\w+", "", tweet)
    return tweet.strip()

model = Detoxify('unbiased')

def detect_hate_speech(tweet):
    cleaned_tweet = clean_tweet(tweet)
    results = model.predict(cleaned_tweet)
    if (results['toxicity'] > 0.7 or
        results['identity_attack'] > 0.7):
        return "Hate Speech"
    return "Not Hate Speech"

results = []
for tweet in tweets:
    label = detect_hate_speech(tweet)
    results.append({"Tweet": tweet, "Label": label})

df = pd.DataFrame(results)

print("Hate Speech Detection Results (Detoxify):")
print(df)

df.to_csv("hate_speech_detoxify_results.csv", index=False)
print("\nResults saved to 'hate_speech_detoxify_results.csv'")

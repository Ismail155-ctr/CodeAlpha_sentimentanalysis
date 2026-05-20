import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

nltk.download('vader_lexicon')

df = pd.read_csv("reviews.csv")

print("\nFIRST 5 ROWS:")
print(df.head())

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

df["Cleaned_Review"] = df["Review"].apply(clean_text)

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):

    score = sia.polarity_scores(text)

    compound = score['compound']

    if compound >= 0.05:
        return "Positive"

    elif compound <= -0.05:
        return "Negative"

    else:
        return "Neutral"

df["Sentiment"] = df["Cleaned_Review"].apply(get_sentiment)

emotion_lexicon = {
    "Happy": ["happy", "good", "great", "excellent", "love", "amazing"],
    "Sad": ["sad", "bad", "disappointed", "poor"],
    "Angry": ["angry", "hate", "worst", "terrible"],
    "Fear": ["fear", "worried", "scared"],
    "Surprise": ["wow", "unexpected", "surprised"]
}

def detect_emotion(text):

    words = text.split()

    for emotion, keywords in emotion_lexicon.items():

        for keyword in keywords:

            if keyword in words:
                return emotion

    return "Neutral"

df["Emotion"] = df["Cleaned_Review"].apply(detect_emotion)

print("\nSENTIMENT COUNTS:")
print(df["Sentiment"].value_counts())

print("\nEMOTION COUNTS:")
print(df["Emotion"].value_counts())


df.to_csv("sentiment_analysis_results.csv", index=False)

print("\nResults saved successfully!")


plt.figure(figsize=(6, 4))

sns.countplot(x=df["Sentiment"])

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(x=df["Emotion"])

plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.show()

all_words = " ".join(df["Cleaned_Review"])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(all_words)

plt.figure(figsize=(10, 5))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title("Most Common Words")

plt.show()

positive_reviews = df[df["Sentiment"] == "Positive"]

negative_reviews = df[df["Sentiment"] == "Negative"]

print("\nTOP POSITIVE REVIEWS:")
print(positive_reviews["Review"].head())

print("\nTOP NEGATIVE REVIEWS:")
print(negative_reviews["Review"].head())

positive_percent = (
    len(positive_reviews) / len(df)
) * 100

negative_percent = (
    len(negative_reviews) / len(df)
) * 100

print("\n============== BUSINESS INSIGHTS ==============")

print(f"Positive Review Percentage: {positive_percent:.2f}%")

print(f"Negative Review Percentage: {negative_percent:.2f}%")

if positive_percent > negative_percent:
    print("Overall customer feedback is positive.")
else:
    print("Product/service improvement is needed.")

print("===============================================")

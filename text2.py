import pandas as pd, re
from textblob import TextBlob
from collections import Counter
data_path = r"C:\Users\yp104\Desktop\ds\amazon_reviews.csv\amazon_reviews.csv"
df = pd.read_csv(data_path)
text = ' '.join(df['reviewText'].dropna().astype(str)).lower()
cleaned = re.sub(r'[^a-z\s]', '', text)
words = cleaned.split()

print("Top 10 words:", Counter(words).most_common(10))
sentiments = df['reviewText'].dropna().apply(lambda x: TextBlob(str(x)).sentiment.polarity)
print("\nSentiment Counts:\n", sentiments.apply(lambda p: "Positive" if p > 0 else "Negative" if p < 0 else "Neutral").value_counts())

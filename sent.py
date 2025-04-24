import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

# Sample Dataset
data = {
    'text': [
        "I love this movie, it's amazing!",
        "Absolutely terrible. I hated it.",
        "What a great performance!",
        "Worst movie I've seen this year.",
        "Not bad, pretty entertaining.",
        "Awful! I walked out halfway.",
        "Fantastic story and direction!",
        "I wouldn’t recommend it to anyone.",
        "Loved every moment of it.",
        "It's boring and too slow."
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative
}

df = pd.DataFrame(data)

# -------------------------
# Text Preprocessing Function
# -------------------------
def process_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply processing
df['clean_text'] = df['text'].apply(process_text)

# Feature Extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# 🔍 Sentiment Analysis Prediction Function
# -------------------------
def predict_sentiment(text):
    processed = process_text(text)
    vect_text = vectorizer.transform([processed])
    prediction = model.predict(vect_text)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    print(f"Input: {text}")
    print(f"Sentiment: {sentiment}")

# -------------------------
# 🔧 Test with your own input
# -------------------------
while True:
    user_input = input("\nEnter a sentence for sentiment analysis (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break
    predict_sentiment(user_input)

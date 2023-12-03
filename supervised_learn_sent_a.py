import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import pandas as pd

# NLTK stopwords
nltk.download('stopwords')

# Sample data
data = {
    "Feedback": ["Great service, I'm very satisfied", 
                 "Poor experience, the product was broken", 
                 "Average service, nothing special",
                 "Loved the product, it was fantastic", 
                 "Terrible customer support"],
    "Sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative"]
}

df = pd.DataFrame(data)

# Preprocessing
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(df['Feedback'])
y = df['Sentiment']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Model Evaluation
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Use the model to analyze new feedback
new_feedback = ["The product was great, but the customer support was terrible"]
new_feedback_transformed = vectorizer.transform(new_feedback)
new_prediction = model.predict(new_feedback_transformed)
print(f"Sentiment of new feedback: {new_prediction[0]}")

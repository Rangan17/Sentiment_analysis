# train_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Example: Training a simple text classifier
def train_and_save_model():
    X = ["I love this!", "This is bad", "I am happy", "I am sad"]
    y = ["positive", "negative", "positive", "negative"]

    vectorizer = TfidfVectorizer()
    X_transformed = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_transformed, y)

    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(model, 'model.pkl')

if __name__ == "__main__":
    train_and_save_model()

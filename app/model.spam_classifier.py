import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_spam_classifier(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit the vectorizer to the training data and transform both sets
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a Multinomial Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model on the testing set
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")

    # Save the trained model and vectorizer
    joblib.dump(model, 'models/spam_classifier.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')

if __name__ == "__main__":
    train_spam_classifier('data/spam_data.csv')
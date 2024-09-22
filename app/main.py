from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

app = FastAPI()

class SpamMessage(BaseModel):
    message: str

# Load the pre-trained model and vectorizer
model = joblib.load('models/spam_classifier.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

@app.post("/classify")
async def classify_spam(message: SpamMessage):
    # Vectorize the input message
    vectorized_message = vectorizer.transform([message.message])

    # Predict the spam classification
    prediction = model.predict(vectorized_message)

    # Return the classification result
    if prediction[0] == 0:
        return {"result": "Not Spam"}
    else:
        return {"result": "Spam"}
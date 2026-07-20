import joblib

model = joblib.load('model/naive_bayes.joblib')
vectorizer = joblib.load('model/tfidf.joblib')

def predict_sentiment(texts):

    X = vectorizer.transform(texts)

    prediction = model.predict(X)

    return prediction
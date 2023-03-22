from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
app = Flask(__name__)
model=load_model('best_model.h5')
with open("tokenizer.pickle" , 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/')

def Home():
    return render_template('in.html')
@app.route("/predict", methods=['POST','GET'])


def predict():
    if request.method == 'POST':
        inputtext = request.form['inputtext']
        text = [inputtext]
        sentiment_classes = ['Negative', 'Neutral', 'Positive']
        max_len=50
        xt = tokenizer.texts_to_sequences(text)
        xt = pad_sequences(xt, padding='post', maxlen=max_len)
        yt = model.predict(xt).argmax(axis=1)
    return render_template('in.html', prediction=sentiment_classes[yt[0]])

if __name__ == "__main__":
    app.run(debug=True)
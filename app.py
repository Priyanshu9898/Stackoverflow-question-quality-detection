from flask import Flask, url_for, redirect, render_template, request
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize 

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
model = tf.keras.models.load_model('my_model.h5')
from keras.preprocessing.text import Tokenizer
max_features = 20000

# tokenizer = Tokenizer(
#     num_words= max_features,
#     lower = True,
#     split=' ',
#     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
#     oov_token = '<OOK>'
# )



app = Flask(__name__)
app.secret_key = 'secret-key'

# def preprocess():
#     preprocessed_question = ""
#     return preprocessed_question

@app.route('/')
def index():    
    return render_template('index.html')

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        question = request.form['que']
        q = question
        # question = text_preprocessing(question)
        
        question = [question]
        tokenizer.fit_on_texts(question)
        question = tokenizer.texts_to_sequences(question)
        
        # question = [list(question)]
        
        # df = pd.DataFrame(question)
        # df = np.array(question)
        df = question
        
        df = pad_sequences(df, maxlen=300)
        
        prediction = model.predict(df)
        
        r = np.argmax(prediction)
        
        if r == 0:
            r = "Low quality"
        elif r == 1:
            r = "Medium quality"
        else:
            r = "High quality"
        # if prediction == 0:
        #     prediction = "Not a duplicate question"
        # else:
        #     prediction = "Duplicate question"
        return render_template('prediction.html', prediction=r)

if __name__ == '__main__':

    app.run(port=8080)








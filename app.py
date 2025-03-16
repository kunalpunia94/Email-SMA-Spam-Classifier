import streamlit as st
import pickle
import string
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure the necessary NLTK data is available
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

ps = PorterStemmer()

# Preprocess function
def transform_test(text):
    text = text.lower()  # 1. Convert to lowercase
    text = word_tokenize(text)  # 2. Tokenization

    # Removing special characters
    y = [i for i in text if i.isalnum()]

    # Removing stopwords & punctuation
    stop_words = set(stopwords.words('english'))  # Ensure stopwords are loaded
    y = [i for i in y if i not in stop_words and i not in string.punctuation]

    # Stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title('Email/SMS Spam Classifier')

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input
    transformed_sms = transform_test(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

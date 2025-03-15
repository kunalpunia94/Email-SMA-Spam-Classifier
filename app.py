import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

#preprocess function
def transform_test(text):
    text = text.lower()   #1.lower case
    text = word_tokenize(text)  #2.Tokenization
    #till here it is converted into list so we can run the loop and remove the special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    #removing the punctutation
    text = y[:] #we have to copy like this
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    #stemmming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    #1. Preprocess  ->function written above
    transformed_sms = transform_test(input_sms)
    #2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    #3. Predict
    result = model.predict(vector_input)[0]
    #4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


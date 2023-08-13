import streamlit as st
import pickle
tfidf = pickle.load(open("./vectorizer.pkl","rb"))
model = pickle.load(open("./model.pkl","rb"))


from nltk.corpus import stopwords
import nltk
stopwords.words("english")
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def text_transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    for i in text:
        a=ps.stem(i)
        y.append(a)
            
    return " ".join(y)


st.title("Email/SMS Smap Classifier")
text=st.text_area("Enter text")
# preprocess
if st.button("predict"):
    transformed=text_transform(text)
    # vectotize
    vect=tfidf.transform([transformed])
    # predict
    res=model.predict(vect)[0]
    # display
    if res==1:
        st.header("Spam")
    else:st.header("not spam")
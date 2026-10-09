<div align="center">

# 📧 Email / SMS Spam Classifier

**Paste a message and find out instantly whether it is spam - NLP preprocessing, TF-IDF and a trained classifier behind a Streamlit app.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-informational)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## ✨ Overview

**Notebook (`a.ipynb`)**

- 📊 EDA on `spam.csv` (word clouds, message length analysis).
- 🧹 Text preprocessing: lower-case → tokenize → remove special characters, stop words and punctuation → **stem** (Porter).
- 🔢 TF-IDF vectorization.
- 🤖 Many classifiers compared (Naive Bayes variants, Logistic Regression, Decision Tree, KNN, Random Forest, AdaBoost, Bagging, Gradient Boosting, XGBoost) - the chosen model reaches about **97 % accuracy**.
- 💾 `model.pkl` and `vectorizer.pkl` saved with pickle.

**App (`st.py`)** - type or paste an email / SMS, press **predict**, and see **Spam** or **not spam**.

## 🚀 Getting Started

```bash
git clone https://github.com/Arashomranpour/email-sms-spam-classifier.git
cd email-sms-spam-classifier
pip install streamlit nltk scikit-learn pandas numpy
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
streamlit run st.py
```

## 📁 Project Structure

```
.
├── a.ipynb          # EDA, preprocessing, model comparison
├── st.py            # Streamlit app
├── model.pkl        # Trained classifier
├── vectorizer.pkl   # TF-IDF vectorizer
└── spam.csv         # Dataset
```

## 🛠️ Tech Stack

`NLTK` · `scikit-learn` · `XGBoost` · `pandas` · `Streamlit`

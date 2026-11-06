Overview

This project is a Machine Learning web application built to classify news text as either "Real" or "Fake". It uses Natural Language Processing techniques and runs on Streamlit. The Smart Fake News Detection System relies on Machine Learning models and Natural Language Processing to separate news from fake news. Streamlit provides a front-end that lets users test the Smart Fake News Detection System directly in the browser.

Files Included

1. train_model.py: Script that creates data, trains the TF-IDF vectorizer and Logistic Regression model, and saves them as .pkl files.

2. app.py: The Streamlit web application interface.

3. requirements.txt: List of Python dependencies needed for the project.

4. my_model.pkl & my_vectorizer.pkl: Saved machine learning model and NLP vectorizer files.

Technologies Used

1. Python: Core programming language.

2. Scikit-Learn: Used for TF-IDF vectorization and Logistic Regression modeling.

3. Pandas: Used for data manipulation.

4. Streamlit: Frontend web framework for deployment.

How to Run Locally

1. Install dependencies with pip install -r requirements.txt.

2. Train the model using python train_model.py.

3. Start the web app with streamlit run app.py.
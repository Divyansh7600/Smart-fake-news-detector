
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle  # using pickle to save our model


texts = [
    "The stock market reached an all-time high today.",
    "Aliens have landed in New York!",
    "The President signed a new healthcare bill.",
    "Drinking bleach will cure diseases.",
    "Scientists discover a new frog.",
    "The earth is flat.",
    "Local team wins the championship.",
    "A man flew to Mars using balloons."
] * 50

labels = [1, 0, 1, 0, 1, 0, 1, 0] * 50


df = pd.DataFrame({'text': texts, 'label': labels})


x_data = df['text']
y_data = df['label']


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


tf_vec = TfidfVectorizer(stop_words='english')
x_train_features = tf_vec.fit_transform(x_train)
x_test_features = tf_vec.transform(x_test)


my_model = LogisticRegression()


my_model.fit(x_train_features, y_train)


preds = my_model.predict(x_test_features)


acc = accuracy_score(y_test, preds)
print("Accuracy of my model is:", acc * 100, "%")


with open('my_model.pkl', 'wb') as f:
    pickle.dump(my_model, f)

with open('my_vectorizer.pkl', 'wb') as f:
    pickle.dump(tf_vec, f)

print("Model saved successfully!")
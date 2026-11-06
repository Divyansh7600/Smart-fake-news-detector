import streamlit as st
import pickle


with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('my_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


st.title("📰 Fake News Detection App")
st.write("Enter your news below to check if it is real or fake.")


user_text = st.text_area("Paste news here:")


if st.button("Check News"):
    if user_text == "":
        st.write("Please type something first.")
    else:
       
        text_features = vectorizer.transform([user_text])
        
        
        result = model.predict(text_features)
        
      
        if result[0] == 1:
            st.success("Result: This news is REAL.")
        else:
            st.error("Result: This news is FAKE.")
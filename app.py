import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import re
import string

# Tokenize function
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r'\1', s).split()

# Load the vectorizer
try:
    vec = joblib.load("model/vectorizer.pkl")
except FileNotFoundError:
    st.error("Vectorizer file not found. Please ensure 'model/vectorizer.pkl' is available.")
    st.stop()

# Load models and 'r' values
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
models = {}
rs = {}

for label in label_cols:
    model_file = f'model/model_{label}.pkl'
    if os.path.exists(model_file):
        model, r = joblib.load(model_file)
        models[label] = model
        rs[label] = r
    else:
        st.error(f"Model file for '{label}' not found. Please ensure '{model_file}' is available.")
        st.stop()

# Streamlit app definition
def main():
    st.title('🚦 Toxic Comment Classifier')
    st.write('Enter a comment below to check its toxicity levels. Only high probability labels will be displayed.')

    # Text input
    comment = st.text_area('Comment', height=150)

    # Add a slider to select the threshold
    threshold = st.slider('Select Toxicity Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    if st.button('Check Toxicity'):
        if comment.strip() == '':
            st.warning("Please enter a comment to analyze.")
        else:
            # Preprocess the comment
            comment_vec = vec.transform([comment])
            predictions = {}

            for label in label_cols:
                r = rs[label]
                model = models[label]
                comment_nb = comment_vec.multiply(r)
                proba = model.predict_proba(comment_nb)[:, 1][0]  # Get the probability for the label
                proba_percent = round(proba * 100, 2)
                if proba >= threshold:
                    predictions[label] = proba_percent

            # Display results based on threshold
            if predictions:
                st.subheader(f"Results (Threshold: {threshold * 100}%):")
                for label, proba in predictions.items():
                    color = "green" if proba < 40 else "orange" if proba < 70 else "red"
                    st.markdown(f"<span style='color:{color}; font-size: 18px;'>**{label.capitalize()}**: <span style='background-color:{color}; color:white; padding: 4px 8px; border-radius: 5px;'>{proba}%</span></span>", unsafe_allow_html=True)
            else:
                st.info(f"No labels exceed the {threshold * 100}% threshold for toxicity.")

            # Display a bar chart for high-probability labels
            if predictions:
                st.subheader("Toxicity Levels")
                chart_data = pd.DataFrame({
                    'Label': list(predictions.keys()),
                    'Probability': list(predictions.values())
                })
                chart_data.set_index('Label', inplace=True)
                st.bar_chart(chart_data)

# Call the main function
if __name__ == '__main__':
    main()

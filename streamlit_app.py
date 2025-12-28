import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
from keras.datasets import imdb

# Page configuration
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="wide")

# Title and description
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("Predict whether a movie review is **Positive** or **Negative** using a SimpleRNN model trained on IMDB data.")

# Load resources
@st.cache_resource
def load_resources():
    # Load word index
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    
    # Load model
    model_path = 'simple_rnn_imdb(1).h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        model_status = f"✓ Model loaded from {model_path}"
    else:
        st.warning(f"Model file '{model_path}' not found. Using untrained model.")
        model_status = "⚠ Using untrained model"
    
    return word_index, reverse_word_index, model, model_status

# Load resources
word_index, reverse_word_index, model, model_status = load_resources()

# Configuration
max_features = 10000
max_len = 500

# Preprocessing function
def preprocess_text(text):
    """Preprocess text using IMDB's word index"""
    words = text.lower().split()
    
    encoded_review = []
    for word in words:
        if word in word_index:
            idx = word_index[word]
            if idx < max_features:
                encoded_review.append(idx + 3)
            else:
                encoded_review.append(2)
        else:
            encoded_review.append(2)
    
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review

# Prediction function
def predict_review(review_text):
    """Predict sentiment for a review"""
    preprocessed = preprocess_text(review_text)
    prediction = model.predict(preprocessed, verbose=0)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    confidence = float(prediction[0][0])
    return sentiment, confidence

# Initialize session state
if 'review_text' not in st.session_state:
    st.session_state.review_text = ""

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(f"{model_status}")
    st.markdown("""
    This app uses a **SimpleRNN** model trained on the IMDB dataset.
    - **Max Features**: 10,000 words
    - **Max Length**: 500 tokens
    - **Architecture**: Embedding → SimpleRNN → Dense
    """)

# Example reviews section (moved to top)
st.subheader("📌 Quick Examples")
col1, col2 = st.columns(2)

with col1:
    if st.button("✨ Try Positive Example", use_container_width=True, key="pos_btn"):
        st.session_state.review_text = "This movie was absolutely amazing! The acting was brilliant, the plot was engaging, and I laughed throughout. Highly recommended!"

with col2:
    if st.button("😞 Try Negative Example", use_container_width=True, key="neg_btn"):
        st.session_state.review_text = "This was the worst movie I've ever seen. Terrible acting, boring plot, and a complete waste of time. I do not recommend it at all."

st.divider()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter a Movie Review")
    review_input = st.text_area(
        "Write your movie review here:",
        value=st.session_state.review_text,
        placeholder="e.g., This movie was absolutely amazing! Great acting and brilliant plot...",
        height=150
    )

with col2:
    st.subheader("🎯 Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Predict button
if st.button("🔮 Predict Sentiment", use_container_width=True, type="primary"):
    if review_input.strip():
        with st.spinner("Analyzing review..."):
            sentiment, confidence = predict_review(review_input)
            
            # Display results
            st.divider()
            st.subheader("📊 Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if sentiment == 'Positive':
                    st.metric("Sentiment", "😊 Positive")
                else:
                    st.metric("Sentiment", "😞 Negative")
            
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            with col3:
                if sentiment == 'Positive':
                    st.metric("Score", f"{confidence:.4f}")
                else:
                    st.metric("Score", f"{1-confidence:.4f}")
            
            # Progress bar
            st.divider()
            st.subheader("Confidence Distribution")
            col1, col2 = st.columns(2)
            with col1:
                st.progress(confidence, f"Positive: {confidence:.2%}")
            with col2:
                st.progress(1-confidence, f"Negative: {(1-confidence):.2%}")
    else:
        st.error("Please enter a movie review!")

# Footer
st.divider()
st.markdown("""
---
**Model Info**: SimpleRNN trained on IMDB dataset | Max Vocabulary: 10,000 | Max Review Length: 500 tokens
""")

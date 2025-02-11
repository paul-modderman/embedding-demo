import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

st.title("Text Embeddings Demo")

# Text input field
input_text = st.text_input("Enter text to generate embeddings:")

# Button to invoke the embeddings API
if st.button("Generate Embeddings"):
    if input_text:
        response = client.embeddings.create(
            input=input_text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        
        # Display the first 20 embedding values
        st.subheader("Embedding Values (first 20):")
        st.write(embedding[:20])
    else:
        st.error("Please enter some text.")
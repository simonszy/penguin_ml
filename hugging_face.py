import streamlit as st
from openai import OpenAI
from transformers import pipeline

st.title("Hugging Face Demo")
text = st.text_input("Enter text to analyze")
model = pipeline("sentiment-analysis")

@st.cache_resource
def load_model():
  return pipeline("sentiment-analysis")
model = load_model()
if text:
  result = model(text)
  st.write(f"Sentiment: {result[0]['label']}")
  st.write(f"Confidence: {result[0]['score']}")
    
st.title("OpenAI Version")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

system_message_default = """You are a helpful sentiment analysis assistant.
You always respond with the sentiment of the text you are given and the confidence of your analysis with a number between 0 and 1
"""
system_message = st.text_area("Enter a System Message to instruct OpenAI", system_message_default) 

analyze_button = st.button("Analyze")

if analyze_button:
  messages = [
    {"role": "system", "content": f"{system_message}",
    },
    {"role": "user", "content": f"Sentiment analysis of the following: {text}"},
  ]
  with st.spinner("Analyzing..."):
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages,)
    
    sentiment = response.choices[0].message.content.strip()
    st.write(sentiment)

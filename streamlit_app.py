import streamlit as st
import fitz
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Set page title
st.set_page_config(page_title="PDF Text Summarizer")

# Load Google T5 model and tokenizer
model_name = 't5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load PDF file
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Define function to extract text from PDF
def extract_text(pdf_file):
    doc = fitz.Document(pdf_file)
    text = ""
    for page in doc:
        text += page.getText()
    return text

# Define function to generate summary
def generate_summary(text):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512)
    summary_ids = model.generate(input_ids, max_length=150, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Generate summary and display it
if pdf_file is not None:
    text = extract_text(pdf_file)
    summary = generate_summary(text)
    st.subheader('Original Text')
    st.write(text)
    st.subheader('Summary')
    st.write(summary)

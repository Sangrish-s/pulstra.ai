import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer,BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import requests
import fitz  # PyMuPDF for handling PDF files
import arvix  # Importing your arxiv.py script
import torch

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the function
load_css("Styles.css")

# Initialize the sentence transformer model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to generate summary with GPT-3.5
def generate_summary(text, language, tokenizer, model):
    template = f"""
###Role:
Your task is to answer the Question by using the Reference and make sure to elaborate about it in Output using the Reference.
Make sure it actually answers the Question and make sure to convey it in {language}.
###Reference:
{text}
###Question: Can you make sure to summarize this Reference for me to conduct a literature review in 150 words and make sure to convey it in {language}
###Output:
    """
    inputs = tokenizer(f"{template}", return_tensors='pt').input_ids.to('cuda:0')
    outputs = model.generate(input_ids=inputs, max_new_tokens=2048, do_sample=True, top_p=1.0, top_k=100, temperature=0.65)
    gen_text = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(template):]
    return gen_text  

def find_closest_papers(user_input, num_matches=1):
    user_embedding = embedding_model.encode(user_input)
    df = pd.read_csv("arxiv_papers_with_embeddings.csv")
    paper_embeddings = df['embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else x).tolist()
    similarities = cosine_similarity([user_embedding], paper_embeddings)[0]
    top_indices = np.argsort(similarities)[-num_matches:]
    return df.iloc[top_indices]

def get_document_text(filename):
    if filename.endswith('.pdf'):
        doc = fitz.open(filename)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    elif filename.endswith('.txt'):
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    else:
        raise ValueError("Unsupported file type. Please provide a PDF or TXT file.")

def download_arxiv_paper(paper_id):
    download_link = f"https://arxiv.org/pdf/{paper_id}.pdf"
    response = requests.get(download_link)
    if response.status_code == 200:
        filename = "Downloaded.pdf"
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename, True
    else:
        return None, False

def generate_answer(question, context, tokenizer, model, document_text, lang):
    role = f"""You are an assistant and a chatbot who is friendly to user and converses with them while also accurately Answers based on Question and Context purely according to the Document. Only according to Question in this language :{lang}. Make sure to return the accurate title and a number with the title as it is mentioned in the document."""
    prompt = f"### Role: {role}\n\n### Document: {document_text}\n\n### Context: {context}\n\n### Question: {question}.in this language :{lang}\n\n### Answers: "
    inputs = tokenizer.encode(prompt, return_tensors='pt').to("cuda:0")
    streamer = TextStreamer(tokenizer)
    outputs = model.generate(inputs, max_new_tokens=8192, streamer=streamer, temperature=0.6, top_p=0.1, top_k=15)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("### Answers:")[1].strip()
    return answer

st.title("Research Paper Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

topic = st.text_input("Enter your research topic:")
language = st.text_input("Enter your preferred language:")

if st.button("Search Papers"):
    if topic and language:
        arvix.main(topic)
        df = pd.read_csv("arxiv_papers_with_embeddings.csv")
        closest_papers = find_closest_papers(topic, 10)

        with open("abstracts.txt", "w", encoding="utf-8") as file:
            counter = 1
            for index, paper in closest_papers.iterrows():
                title = paper['title']
                abstract = paper['abstract']
                file.write(f"{counter}: Paper Title: {title}\n")
                file.write("Abstract:\n")
                file.write(f"{abstract}\n\n")
                counter += 1

        st.session_state.closest_papers = closest_papers
        st.session_state.document_text = get_document_text("abstracts.txt")
        st.session_state.original_document = st.session_state.document_text

if "closest_papers" in st.session_state:
    st.write("Closest Papers:")
    for i, paper in st.session_state.closest_papers.iterrows():
        st.write(f"{i+1}: {paper['title']}")

user_input = st.text_input("You: ")
if st.button("Send"):
    if user_input.isdigit():
        paper_number = int(user_input)
        if 1 <= paper_number <= len(st.session_state.closest_papers):
            paper_index = paper_number - 1
            paper_id = st.session_state.closest_papers.iloc[paper_index]['id']
            filename, downloaded = download_arxiv_paper(paper_id)
            if downloaded:
                st.session_state.document_text = get_document_text(filename)
                st.session_state.chat_history.append({"user": f"Switched to paper {paper_number}"})
            else:
                st.session_state.chat_history.append({"bot": "Failed to download the requested paper."})
        else:
            st.session_state.chat_history.append({"bot": "Invalid paper number."})
    elif user_input.lower() == "back":
        st.session_state.document_text = st.session_state.original_document
        st.session_state.chat_history.append({"user": "Switched back to the original document."})
    else:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map='cuda:0')
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=bnb_config, device_map='cuda:0')
        context = "".join([msg["user"] + msg.get("bot", "") for msg in st.session_state.chat_history])
        answer = generate_answer(user_input, context, tokenizer, model, st.session_state.document_text, language)
        st.session_state.chat_history.append({"user": user_input, "bot": answer})

if st.button("Clear Chat History"):
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if "user" in message:
        st.markdown(f"**You:** {message['user']}")
    if "bot" in message:
        st.markdown(f"**Bot:** {message['bot']}")



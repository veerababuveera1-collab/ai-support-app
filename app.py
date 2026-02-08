import streamlit as st
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load API key
load_dotenv()

st.set_page_config(page_title="AI Support System", layout="wide")
st.title("🤖 Automated AI Customer Support")

# Initialize vector DB
if "db" not in st.session_state:
    st.session_state.db = None

# Sidebar: Document upload
st.sidebar.header("📄 Upload Knowledge Document")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(uploaded_file.name)
    else:
        loader = TextLoader(uploaded_file.name)

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    st.session_state.db = db
    st.sidebar.success("Document loaded into memory!")

# Intent detection
def detect_intent(message):
    msg = message.lower()

    if "password" in msg or "login" in msg:
        return "Account Issue"
    elif "refund" in msg or "charged" in msg:
        return "Billing Issue"
    elif "price" in msg or "plan" in msg:
        return "Sales Inquiry"
    elif "not working" in msg or "error" in msg:
        return "Technical Issue"
    else:
        return "General Inquiry"

# Automation logic
def automation_action(intent):
    if intent == "Account Issue":
        return "Password reset link sent."
    elif intent == "Billing Issue":
        return "Refund ticket created."
    elif intent == "Technical Issue":
        return "Support ticket created."
    elif intent == "Sales Inquiry":
        return "Sales team notified."
    else:
        return "No automation required."

# Chat interface
st.subheader("💬 Chat with AI Support")
user_input = st.text_input("Type your message:")

if user_input:
    intent = detect_intent(user_input)

    # If document memory exists
    if st.session_state.db:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.db.as_retriever()
        )
        answer = qa.run(user_input)
    else:
        llm = ChatOpenAI(model="gpt-4o-mini")
        answer = llm.invoke(user_input).content

    action = automation_action(intent)

    st.markdown("### 🧠 AI Analysis")
    st.write(f"**Intent:** {intent}")

    st.markdown("### 🤖 AI Response")
    st.write(answer)

    st.markdown("### ⚙️ Automation Action")
    st.success(action)

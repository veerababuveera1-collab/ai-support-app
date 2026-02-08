import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load env
load_dotenv()

st.set_page_config(page_title="AI Support System", layout="wide")
st.title("🤖 Automated AI Customer Support")

# ---------------------------
# Model selection
# ---------------------------
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select AI Provider",
    ["OpenRouter", "OpenAI"]
)

# ---------------------------
# Initialize vector DB
# ---------------------------
if "db" not in st.session_state:
    st.session_state.db = None

# ---------------------------
# Sidebar: Document upload
# ---------------------------
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

    # Free local embeddings (no OpenAI billing)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    st.session_state.db = db
    st.sidebar.success("Document loaded into memory!")

# ---------------------------
# Intent detection
# ---------------------------
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

# ---------------------------
# Automation logic
# ---------------------------
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

# ---------------------------
# Chat functions
# ---------------------------
def openrouter_chat(message):
    api_key = st.secrets["OPENROUTER_API_KEY"]

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openrouter/auto",
            "messages": [
                {"role": "user", "content": message}
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]


def openai_chat(message):
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}]
    )

    return response.choices[0].message.content

# ---------------------------
# Chat interface
# ---------------------------
st.subheader("💬 Chat with AI Support")
user_input = st.text_input("Type your message:")

if user_input:
    intent = detect_intent(user_input)

    # Retrieve context if available
    if st.session_state.db:
        docs = st.session_state.db.similarity_search(user_input, k=3)
        context = "\n".join([d.page_content for d in docs])
        prompt = f"Answer using this context:\n{context}\n\nQuestion: {user_input}"
    else:
        prompt = user_input

    # Model switch
    if model_choice == "OpenRouter":
        answer = openrouter_chat(prompt)
    else:
        answer = openai_chat(prompt)

    action = automation_action(intent)

    st.markdown("### 🧠 AI Analysis")
    st.write(f"**Intent:** {intent}")

    st.markdown("### 🤖 AI Response")
    st.write(answer)

    st.markdown("### ⚙️ Automation Action")
    st.success(action)

import streamlit as st
import requests
import uuid
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

st.set_page_config(page_title="VeeraTech AI Support", layout="wide")
st.title("🤖 VeeraTech AI Customer Support")

# ---------------------------
# Chat memory
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# Built-in Knowledge Base
# ---------------------------
KB_RESPONSES = {
    "Account Issue":
        "If your account is locked, reset your password here:\n"
        "https://veeratech.ai/reset\n\n"
        "If the issue continues, contact support@veeratech.ai.",

    "Billing Issue":
        "Refunds are available within 7 days of purchase.\n"
        "Email support@veeratech.ai with your order ID.",

    "Sales Inquiry":
        "Pricing Plans:\n"
        "- Basic: $10/month – AI chat support\n"
        "- Pro: $29/month – AI chat + automation\n"
        "- Enterprise: Custom pricing",

    "Technical Issue":
        "Try the following steps:\n"
        "1. Refresh the page\n"
        "2. Clear browser cache\n"
        "3. Restart the application\n\n"
        "If the issue continues, contact support@veeratech.ai.",

    "Contact":
        "Support Email: support@veeratech.ai\n"
        "Support Hours: Mon–Fri, 9 AM – 6 PM IST\n"
        "WhatsApp: +91-90000-00000",

    "Account Deletion":
        "To delete your account, email:\n"
        "privacy@veeratech.ai"
}

# ---------------------------
# Model selection
# ---------------------------
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select AI Provider",
    ["OpenRouter", "OpenAI"]
)

# ---------------------------
# Vector DB setup
# ---------------------------
if "db" not in st.session_state:
    st.session_state.db = None

st.sidebar.header("📄 Upload Knowledge Document")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader(uploaded_file.name) if uploaded_file.name.endswith(".pdf") else TextLoader(uploaded_file.name)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    st.session_state.db = db
    st.sidebar.success("Document loaded successfully!")

# ---------------------------
# Intent detection
# ---------------------------
def detect_intent(message: str):
    msg = message.lower()

    # Specific intents first
    if any(w in msg for w in ["delete", "remove account", "close account"]):
        return "Account Deletion", 0.95

    elif any(w in msg for w in ["refund", "charged", "billing", "payment"]):
        return "Billing Issue", 0.90

    elif any(w in msg for w in ["price", "plan", "cost", "pricing"]):
        return "Sales Inquiry", 0.88

    elif any(w in msg for w in ["not working", "error", "bug", "issue"]):
        return "Technical Issue", 0.85

    elif any(w in msg for w in ["contact", "support", "email", "whatsapp"]):
        return "Contact", 0.85

    elif any(w in msg for w in ["password", "login", "locked"]):
        return "Account Issue", 0.90

    return "General Inquiry", 0.60

# ---------------------------
# Automation logic
# ---------------------------
def automation_action(intent: str):
    actions = {
        "Account Issue": "Password reset link provided.",
        "Billing Issue": "Refund instructions sent.",
        "Technical Issue": "Support steps provided.",
        "Sales Inquiry": "Pricing details shared.",
        "Contact": "Support contact details provided.",
        "Account Deletion": "Privacy request instructions sent."
    }
    return actions.get(intent, "No automation required.")

# ---------------------------
# AI Chat functions
# ---------------------------
def openrouter_chat(message: str):
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        return "OpenRouter API key not configured."

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openrouter/auto",
            "messages": [
                {"role": "system", "content": "You are VeeraTech AI support."},
                {"role": "user", "content": message}
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]


def openai_chat(message: str):
    from openai import OpenAI
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API key not configured."

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are VeeraTech AI support."},
            {"role": "user", "content": message}
        ]
    )

    return response.choices[0].message.content

# ---------------------------
# Chat UI
# ---------------------------
st.subheader("💬 Chat with AI Support")
user_input = st.text_input("Type your message:")

if user_input:
    intent, confidence = detect_intent(user_input)
    source = "AI"

    # Ticket ID generation
    ticket_id = f"VT-{str(uuid.uuid4())[:8].upper()}"

    # Direct KB response
    if intent in KB_RESPONSES:
        answer = KB_RESPONSES[intent]
        source = "Knowledge Base"
    else:
        if st.session_state.db:
            docs = st.session_state.db.similarity_search(user_input, k=3)
            context = "\n".join([d.page_content for d in docs])
            prompt = f"Context:\n{context}\n\nQuestion:\n{user_input}"
        else:
            prompt = user_input

        if model_choice == "OpenRouter":
            answer = openrouter_chat(prompt)
        else:
            answer = openai_chat(prompt)

    action = automation_action(intent)

    # Save to chat history
    st.session_state.chat_history.append({
        "user": user_input,
        "intent": intent,
        "confidence": confidence,
        "answer": answer,
        "action": action,
        "ticket": ticket_id
    })

# ---------------------------
# Display chat history
# ---------------------------
for chat in reversed(st.session_state.chat_history):
    st.markdown("---")
    st.write(f"**User:** {chat['user']}")
    st.write(f"**Intent:** {chat['intent']} ({chat['confidence']*100:.0f}% confidence)")
    st.write(f"**Response:** {chat['answer']}")
    st.write(f"**Automation:** {chat['action']}")
    st.caption(f"Ticket ID: {chat['ticket']}")

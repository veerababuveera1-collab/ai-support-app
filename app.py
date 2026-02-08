import streamlit as st
import requests, os, sqlite3, bcrypt
from datetime import datetime
import matplotlib.pyplot as plt
import re

# ======================
# DATABASE SETUP
# ======================

conn = sqlite3.connect("autosolve.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password BLOB,
    tenant_id INTEGER
)""")

c.execute("""CREATE TABLE IF NOT EXISTS workflows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id INTEGER,
    name TEXT,
    description TEXT,
    created_at TEXT
)""")

c.execute("""CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id INTEGER,
    event_type TEXT,
    event_data TEXT,
    created_at TEXT
)""")

conn.commit()

# ======================
# AUTH
# ======================

def create_user(username, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    tenant_id = abs(hash(username)) % 10000
    try:
        c.execute(
            "INSERT INTO users (username, password, tenant_id) VALUES (?, ?, ?)",
            (username, hashed, tenant_id),
        )
        conn.commit()
        return True
    except:
        return False

def authenticate(username, password):
    c.execute("SELECT password, tenant_id FROM users WHERE username=?", (username,))
    row = c.fetchone()
    if row and bcrypt.checkpw(password.encode(), row[0]):
        return row[1]
    return None

def login_ui():
    st.subheader("🔐 AutoSolve Login")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            tenant = authenticate(u, p)
            if tenant:
                st.session_state.user = u
                st.session_state.tenant_id = tenant
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if create_user(u, p):
                st.success("Account created")
            else:
                st.error("Username exists")

if "user" not in st.session_state:
    login_ui()
    st.stop()

tenant_id = st.session_state.tenant_id

# ======================
# AI CONFIG
# ======================

API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"  # fast model

def call_ai(prompt):
    if not API_KEY:
        return "Missing OPENROUTER_API_KEY"

    try:
        r = requests.post(
            URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "AI service unavailable"

# ======================
# PROMPT SAFETY
# ======================

danger_patterns = [
    r'password',
    r'secret',
    r'api_key',
    r'drop table',
    r'delete database'
]

def is_safe(prompt):
    for pattern in danger_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return False
    return True

# ======================
# AUDIT LOG
# ======================

def log_event(event_type, data):
    c.execute(
        "INSERT INTO audit_log VALUES (NULL,?,?,?,?)",
        (tenant_id, event_type, data, datetime.now().isoformat())
    )
    conn.commit()

# ======================
# UI
# ======================

st.set_page_config("AutoSolve", layout="wide")
st.title("⚙️ AutoSolve – Enterprise AI Workflow Platform")

tabs = st.tabs([
    "💬 AI Assistant",
    "⚙️ Create Workflow",
    "📁 My Workflows",
    "📊 Analytics",
    "📜 Audit Log"
])

# -------- AI ASSISTANT --------
with tabs[0]:
    q = st.chat_input("Ask AutoSolve...")
    if q:
        if not is_safe(q):
            st.error("Prompt blocked for security reasons.")
        else:
            ans = call_ai(q)
            st.write(ans)
            log_event("ai_query", q)

# -------- CREATE WORKFLOW --------
with tabs[1]:
    name = st.text_input("Workflow name")
    desc = st.text_area("Describe automation or AI workflow")

    if st.button("Generate Workflow"):
        if not is_safe(desc):
            st.error("Unsafe workflow description.")
        else:
            result = call_ai(
                f"Design an enterprise automation workflow:\n{desc}"
            )
            c.execute(
                "INSERT INTO workflows VALUES (NULL,?,?,?,?)",
                (tenant_id, name, result, datetime.now().isoformat())
            )
            conn.commit()
            log_event("workflow_created", name)
            st.success("Workflow created")
            st.write(result)

# -------- MY WORKFLOWS --------
with tabs[2]:
    c.execute(
        "SELECT id,name,created_at FROM workflows WHERE tenant_id=?",
        (tenant_id,)
    )
    for wid, name, created in c.fetchall():
        if st.button(f"{name} ({created})", key=wid):
            c.execute("SELECT description FROM workflows WHERE id=?", (wid,))
            st.write(c.fetchone()[0])

# -------- ANALYTICS --------
with tabs[3]:
    c.execute("SELECT COUNT(*) FROM workflows WHERE tenant_id=?", (tenant_id,))
    wf_count = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM audit_log WHERE tenant_id=?", (tenant_id,))
    log_count = c.fetchone()[0]

    st.metric("Workflows", wf_count)
    st.metric("Audit Events", log_count)

    fig, ax = plt.subplots()
    ax.bar(["Workflows", "Audit Events"], [wf_count, log_count])
    st.pyplot(fig)

# -------- AUDIT LOG --------
with tabs[4]:
    c.execute(
        "SELECT event_type,event_data,created_at FROM audit_log "
        "WHERE tenant_id=? ORDER BY id DESC LIMIT 20",
        (tenant_id,)
    )
    logs = c.fetchall()
    for e, d, t in logs:
        st.write(f"[{t}] {e} → {d}")

# -------- LOGOUT --------
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

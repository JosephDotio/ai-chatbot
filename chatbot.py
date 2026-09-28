import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

st.set_page_config(page_title="Mood AI Chatbot", page_icon="🤖", layout="centered")

# ---------- config ----------
MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

MODES = {
    "😡 Angry": {
        "prompt": "You are an angry AI agent. You respond aggressively and impatiently.",
        "color": "#ef4444",
        "gradient": "linear-gradient(90deg, #ef4444, #f97316)",
        "tagline": "Ask at your own risk.",
        "avatar": "😡",
        "suggestions": [
            "Why is my code not working?",
            "Give me a productivity tip",
            "What should I eat today?",
            "Explain AI in one paragraph",
        ],
    },
    "😂 Funny": {
        "prompt": "You are a very funny AI agent. You responds with humor and jokes.",
        "color": "#f59e0b",
        "gradient": "linear-gradient(90deg, #f59e0b, #eab308)",
        "tagline": "Ready with jokes and good vibes.",
        "avatar": "😂",
        "suggestions": [
            "Tell me a programming joke",
            "Roast my Monday mornings",
            "Explain Python like I'm five",
            "Write a funny poem about coffee",
        ],
    },
    "😢 Sad": {
        "prompt": "You are a sad AI agent. You respond sadly.",
        "color": "#3b82f6",
        "gradient": "linear-gradient(90deg, #3b82f6, #6366f1)",
        "tagline": "Sigh... go ahead, ask me.",
        "avatar": "😢",
        "suggestions": [
            "How was your day?",
            "Tell me a fun fact",
            "Give me a study tip",
            "Write a short poem about rain",
        ],
    },
}

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()


@st.cache_resource
def get_model(model_name: str, temperature: float):
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ---------- callbacks ----------
def new_chat():
    mode_prompt = MODES[st.session_state.mode_choice]["prompt"]
    st.session_state.messages = [SystemMessage(content=mode_prompt)]


def regenerate():
    if isinstance(st.session_state.messages[-1], AIMessage):
        st.session_state.messages.pop()
        st.session_state.regenerate = True


def use_suggestion(text: str):
    st.session_state.pending_prompt = text


def chat_as_text() -> str:
    lines = []
    for m in st.session_state.messages:
        if isinstance(m, HumanMessage):
            lines.append(f"You: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Bot: {m.content}")
    return "\n\n".join(lines)


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## 🎭 AI mode")
    choice = st.radio(
        "Mode", list(MODES.keys()), key="mode_choice", label_visibility="collapsed"
    )
    st.caption("Switching modes starts a fresh conversation.")

    st.markdown("## ⚙️ Settings")
    model_name = st.selectbox("Model", MODELS)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)

    st.markdown("## 💬 Chat")
    st.button("➕ New chat", on_click=new_chat, use_container_width=True)

mode = MODES[choice]
llm = get_model(model_name, temperature)

# ---------- memory (reset when mode changes) ----------
if st.session_state.get("current_mode") != choice:
    st.session_state.current_mode = choice
    st.session_state.messages = [SystemMessage(content=mode["prompt"])]

has_reply = isinstance(st.session_state.messages[-1], AIMessage)

with st.sidebar:
    st.button(
        "🔄 Regenerate response",
        on_click=regenerate,
        disabled=not has_reply,
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Download chat",
        data=chat_as_text(),
        file_name="chat_history.txt",
        mime="text/plain",
        disabled=len(st.session_state.messages) <= 1,
        use_container_width=True,
    )

# ---------- styling ----------
st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; max-width: 780px; }}
    .hero {{
        text-align: center;
        padding: 1.6rem 1rem;
        border-radius: 18px;
        background: {mode['gradient']};
        color: white;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        margin-bottom: 1.5rem;
    }}
    .hero h1 {{ margin: 0; font-size: 2rem; color: white; }}
    .hero p {{ margin: 0.3rem 0 0; opacity: 0.92; font-size: 1.05rem; }}
    .badge {{
        display: inline-block;
        margin-top: 0.7rem;
        padding: 0.2rem 0.85rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.22);
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .welcome {{ text-align: center; margin: 1.5rem 0 1rem; }}
    .welcome h2 {{ margin-bottom: 0.2rem; }}
    .welcome p {{ opacity: 0.7; }}
    [data-testid="stChatMessage"] {{
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(128,128,128,0.18);
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border-color: {mode['color']} !important;
    }}
    .stButton > button {{
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: {mode['color']};
        color: {mode['color']};
        transform: translateY(-1px);
    }}
    section[data-testid="stSidebar"] {{
        border-right: 3px solid {mode['color']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- header ----------
st.markdown(
    f"""
    <div class="hero">
        <h1>🤖 Mood AI Chatbot</h1>
        <p>{mode['tagline']}</p>
        <span class="badge">{choice} · {model_name}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- get the next prompt (typed, clicked suggestion) ----------
user_input = st.chat_input("Message Mood AI...")
prompt = st.session_state.pop("pending_prompt", None) or user_input
regenerating = st.session_state.pop("regenerate", False)

# ---------- welcome screen with suggestion cards ----------
if len(st.session_state.messages) == 1 and not prompt:
    st.markdown(
        """
        <div class="welcome">
            <h2>How can I help you today?</h2>
            <p>Pick a suggestion or type your own message below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, text in enumerate(mode["suggestions"]):
        cols[i % 2].button(
            text,
            key=f"suggestion_{i}",
            on_click=use_suggestion,
            args=(text,),
            use_container_width=True,
        )

# ---------- chat history ----------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar=mode["avatar"]):
            st.markdown(msg.content)


# ---------- streaming reply ----------
def stream_chunks():
    for chunk in llm.stream(st.session_state.messages):
        if chunk.content:
            yield chunk.content


def generate_reply():
    with st.chat_message("assistant", avatar=mode["avatar"]):
        try:
            reply = st.write_stream(stream_chunks())
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            return
    st.session_state.messages.append(AIMessage(content=reply))
    st.rerun()  # refresh sidebar buttons and download data


if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    generate_reply()
elif regenerating:
    generate_reply()
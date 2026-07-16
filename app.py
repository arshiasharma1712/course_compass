import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Page config — this sets the browser tab title and icon
st.set_page_config(
    page_title="Course Compass",
    page_icon="🧭",
    layout="centered"
)

@st.cache_resource
def get_collection():
    """
    @st.cache_resource means this function only runs ONCE
    and caches the result. Without this, it would reconnect
    to ChromaDB on every single message — very slow.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "chromadb_store")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(
        name="course_compass",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    return collection


def retrieve_chunks(collection, question, n_results=8):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results['documents'][0]


def get_answer(question, chunks):
    context = "\n\n".join(chunks)
    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are Course Compass, an AI assistant for Texas A&M University students.
You help students make informed decisions about courses and professors using real TAMU grade data.
Always base your answers only on the provided data. Be specific with numbers and statistics.
If data is insufficient, say so honestly. Keep answers to 3-5 sentences."""
            },
            {
                "role": "user",
                "content": f"""Real TAMU grade data:
{context}

Student question: {question}"""
            }
        ],
        max_tokens=1024,
        temperature=0.3
    )
    return response.choices[0].message.content


# ---- UI STARTS HERE ----

st.title("🧭 Course Compass")
st.caption("AI-powered course planning for Texas A&M students")

# Example questions to help users get started
with st.expander("💡 Example questions you can ask"):
    st.markdown("""
    - Who is the best professor for CSCE 221?
    - Which section of MATH 151 has the highest A rate?
    - What are the easiest CSCE electives?
    - Compare professors for CSCE 314
    - Which PHYS 206 professor gives the most A grades?
    """)

# Initialize chat history in session state
# session_state persists data across reruns of the app
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input at the bottom
if prompt := st.chat_input("Ask about TAMU courses or professors..."):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching TAMU data..."):
            collection = get_collection()
            chunks = retrieve_chunks(collection, prompt)
            answer = get_answer(prompt, chunks)
        st.markdown(answer)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
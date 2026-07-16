import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


def get_collection():
    """
    Connect to our existing ChromaDB database.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "chromadb_store")
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(
        name="course_compass",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    
    print(f"Connected to ChromaDB — {collection.count()} chunks available")
    return collection


def retrieve_relevant_chunks(collection, question, n_results=8):
    """
    Step 1 — RETRIEVE.
    Convert the question to an embedding and find the 
    most similar chunks in our database.
    """
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results['documents'][0]


def build_prompt(question, chunks):
    """
    Step 2 — AUGMENT.
    Build a prompt that gives the LLM real TAMU data
    alongside the student's question.
    """
    context = "\n\n".join(chunks)
    
    prompt = f"""You are Course Compass, an AI assistant for Texas A&M University students.
You help students make informed decisions about courses and professors.

Below is real TAMU grade distribution data retrieved for this question.
ONLY use this data to answer. Do not make up information not present in the data.
If the data doesn't contain enough information, say so honestly.

--- TAMU DATA ---
{context}
--- END DATA ---

Student question: {question}

Provide a helpful, specific answer based on the data above.
When mentioning professors, include their grade distribution stats.
When comparing professors, be specific about A rates and GPA averages.
Keep your answer concise but informative — 3-5 sentences is ideal."""
    
    return prompt


def generate_answer(prompt):
    """
    Step 3 — GENERATE.
    Send the augmented prompt to Groq's LLM and get back
    a natural language answer.
    
    We use llama-3.3-70b-versatile — a powerful open source model
    that Groq runs for free. It's excellent at following instructions
    and reasoning over structured data like our grade distributions.
    """
    client = Groq()
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are Course Compass, a helpful AI assistant for Texas A&M University students. You answer questions about courses, professors, and grade distributions using real TAMU data."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1024,
        temperature=0.3
    )
    
    return response.choices[0].message.content


def answer_question(collection, question):
    """
    The main RAG function — combines all 3 steps.
    This is what Streamlit will call when a student asks a question.
    """
    # Step 1: Retrieve
    chunks = retrieve_relevant_chunks(collection, question)
    
    # Step 2: Augment
    prompt = build_prompt(question, chunks)
    
    # Step 3: Generate
    answer = generate_answer(prompt)
    
    return answer


def main():
    collection = get_collection()
    
    test_questions = [
        "Who is the best professor for CSCE 221?",
        "Which professor gives the most A grades in CSCE 315?",
        "What is the easiest computer science course at TAMU?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        answer = answer_question(collection, question)
        print(f"\nA: {answer}")
        print('='*60)


if __name__ == "__main__":
    main()



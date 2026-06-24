# Course Compass 🧭

> AI-powered course planning chatbot for Texas A&M University students

## What it does

Course Compass lets TAMU students ask natural language questions about courses, professors, and degree requirements — and get answers grounded in real data.

**Example questions students can ask:**
- "Who is the best professor for ACCT 229?"
- "What courses are required for a Psychology major?"
- "What classes do I need for the Business Analytics certificate?"
- "What are the prerequisites for CSCE 420?"
- "Which section of CHEM 107 has the highest A rate?"

## Why this is different

Most course planning tools make you dig through PDFs, Rate My Professor, and Howdy separately. Course Compass combines all three into one conversational interface backed by real TAMU data.

## Tech stack

| Tool | Purpose |
|------|---------|
| Python | Primary language |
| LangChain | RAG pipeline framework |
| ChromaDB | Vector database for semantic search |
| Anthropic Claude API | LLM for generating answers |
| Streamlit | Web interface |
| pandas | Data processing |

## Data sources

- **RateMyProfessor** — Professor ratings and reviews (via unofficial API)
- **TAMU Grade Distributions** — Public grade data by course, section, and professor
- **TAMU Course Catalog** — Degree requirements, prerequisites, and track information

## How it works
Student question

↓

LangChain retriever searches ChromaDB

↓

Finds relevant chunks (professor ratings + grade data + catalog info)

↓

Claude API generates a grounded, accurate answer

↓

Streamlit displays response in chat UI

This architecture is called **Retrieval-Augmented Generation (RAG)** — the LLM only answers based on retrieved real data, not hallucination.

## Project structure
course-compass/

├── data/

│   ├── raw/          # Original downloaded data

│   └── processed/    # Cleaned data ready for ChromaDB

├── src/

│   ├── ingest_grades.py      # Processes TAMU grade distribution CSVs

│   ├── ingest_professors.py  # Fetches RateMyProfessor data

│   ├── ingest_catalog.py     # Parses TAMU course catalog

│   ├── vector_store.py       # Loads data into ChromaDB

│   └── rag_chain.py          # LangChain + Claude API wiring

├── app.py            # Streamlit web app

├── requirements.txt  # Dependencies

└── .env.example      # Environment variable template

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/arshiasharma1712/course-compass.git
cd course-compass
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the root directory:
ANTHROPIC_API_KEY=your_key_here

**5. Run the app**
```bash
streamlit run app.py
```

## Status

🚧 **Currently in active development**

- [x] Project structure and environment setup
- [ ] TAMU grade distribution ingestion
- [ ] RateMyProfessor data ingestion
- [ ] Course catalog ingestion
- [ ] ChromaDB vector store setup
- [ ] LangChain RAG pipeline
- [ ] Streamlit chat interface
- [ ] Deployment

## Built by

Arshia Sharma — CS student at Texas A&M University  
[GitHub](https://github.com/arshiasharma1712)

---

*Built for TAMU students, by a TAMU student.*
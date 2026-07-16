import chromadb
import os


def load_chunks(filepath):
    """
    Read our processed text chunks from the .txt file.
    Each line is one chunk — we just read them all into a list.
    """
    print(f"Loading chunks from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        # readlines() gives us a list where each item is one line
        # .strip() removes the newline character (\n) at the end of each line
        chunks = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks, collection_name="course_compass"):
    """
    This is the core function — it takes our text chunks and loads
    them into ChromaDB as searchable embeddings.
    
    Key concept: ChromaDB needs three things for each item:
    - documents: the actual text
    - ids: a unique identifier for each chunk (we just use "chunk_0", "chunk_1", etc.)
    - ChromaDB handles the embedding automatically using a built-in model
    """
    print("\nInitializing ChromaDB...")
    
    # PersistentClient saves the database to disk so it survives
    # between runs — without this, it would reset every time
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "chromadb_store")
    
    client = chromadb.PersistentClient(path=db_path)
    
    # A "collection" is like a table in a regular database
    # get_or_create means: use it if it exists, create it if it doesn't
    # We delete and recreate to start fresh each time we run this
    try:
        client.delete_collection(name=collection_name)
        print("Cleared existing collection")
    except:
        pass  # No collection existed yet, that's fine
    
    collection = client.create_collection(
        name=collection_name,
        # This tells ChromaDB to use cosine similarity when searching
        # Cosine similarity measures the angle between two embedding vectors
        # It's better than raw distance for text similarity
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Created collection: {collection_name}")
    
    # ChromaDB works best when we add documents in batches
    # rather than one at a time — 500 at a time is a good balance
    BATCH_SIZE = 500
    total = len(chunks)
    
    print(f"\nLoading {total} chunks into ChromaDB in batches of {BATCH_SIZE}...")
    print("(This will take a few minutes — it's embedding every chunk)\n")
    
    for i in range(0, total, BATCH_SIZE):
        # Get the current batch
        batch = chunks[i:i + BATCH_SIZE]
        
        # Create unique IDs for each chunk: "chunk_0", "chunk_1", etc.
        ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
        
        # Add to ChromaDB — it automatically converts text to embeddings
        collection.add(
            documents=batch,
            ids=ids
        )
        
        # Progress indicator so you can see it's working
        progress = min(i + BATCH_SIZE, total)
        print(f"  Progress: {progress}/{total} chunks loaded ({round(progress/total*100)}%)")
    
    print(f"\nDone! All {total} chunks loaded into ChromaDB")
    return collection


def test_query(collection):
    """
    Let's immediately test that our vector store actually works
    by asking it a real question a TAMU student might ask.
    
    This is the moment where it becomes real — natural language
    search over 115k rows of data.
    """
    print("\n--- TESTING VECTOR SEARCH ---")
    
    test_questions = [
        "Who is the best professor for CSCE 221?",
        "What is the easiest computer science course?",
        "Which professor gives the most A grades in engineering?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: '{question}'")
        print("Top 3 results:")
        
        # query_texts takes our natural language question,
        # converts it to an embedding, and finds the closest chunks
        results = collection.query(
            query_texts=[question],
            n_results=3  # Return top 3 most relevant chunks
        )
        
        # results['documents'] is a list of lists — [0] gets the first query's results
        for j, doc in enumerate(results['documents'][0]):
            print(f"  {j+1}. {doc[:120]}...")  # Print first 120 characters


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load grade chunks — our main data source
    grade_chunks_path = os.path.join(base_dir, 'data', 'processed', 'grade_chunks.txt')
    chunks = load_chunks(grade_chunks_path)
    
    # Build the vector store
    collection = build_vector_store(chunks)
    
    # Test it immediately with real questions
    test_query(collection)
    
    print("\n✓ Vector store built successfully!")
    print(f"✓ Database saved to: chromadb_store/")
    print("✓ Ready for Phase 4 — RAG pipeline")


if __name__ == "__main__":
    main()
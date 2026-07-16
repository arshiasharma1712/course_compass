import chromadb
import os


def build_vector_store(chunks, collection_name="course_compass"):
    print("\nInitializing ChromaDB...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "chromadb_store")
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(name=collection_name)
        print("Cleared existing collection")
    except:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    BATCH_SIZE = 500
    total = len(chunks)
    print(f"\nLoading {total} chunks into ChromaDB...\n")
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
        collection.add(documents=batch, ids=ids)
        progress = min(i + BATCH_SIZE, total)
        print(f"  Progress: {progress}/{total} ({round(progress/total*100)}%)")
    print(f"\nDone! {total} chunks loaded")
    return collection


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grade_chunks_path = os.path.join(base_dir, 'data', 'processed', 'grade_chunks.txt')

    # Load ALL 115k chunks — every course, every college
    print("Loading all chunks...")
    with open(grade_chunks_path, 'r', encoding='utf-8') as f:
        all_chunks = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(all_chunks)} chunks")

    collection = build_vector_store(all_chunks)
    print("\n✓ Done! Ready for Streamlit UI")


if __name__ == "__main__":
    main()
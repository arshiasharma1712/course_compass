import RateMyProfessor_Database_APIs as rmp
import os

def fetch_all_tamu_professors():
    """
    This approach is smarter than searching one professor at a time.
    RateMyProfessor assigns every school a unique ID.
    TAMU's ID is 1003 — we can see this in the URL:
    ratemyprofessors.com/school/1003
    
    We pass that ID and get back EVERY professor at TAMU in one call.
    This is called a bulk fetch — much faster and more reliable.
    """
    print("Fetching all TAMU professors from RateMyProfessor...")
    print("(This may take 30-60 seconds — it's downloading all professors at once)\n")
    
    TAMU_SCHOOL_ID = '1003'
    professors = rmp.fetch_all_professors_from_a_school(TAMU_SCHOOL_ID)
    
    print(f"\nFetched {len(professors)} professors from RateMyProfessor")
    return professors


def convert_to_text_chunks(professors):
    """
    Convert each professor object into a natural language sentence.
    Same concept as grade chunks — ChromaDB searches text, not objects.
    """
    chunks = []
    skipped = 0
    
    for prof in professors:
        try:
            # Skip professors with no ratings — not useful for students
            if not hasattr(prof, 'rating') or prof.rating is None:
                skipped += 1
                continue
            
            # Some professors don't have all fields — we handle that gracefully
            name = getattr(prof, 'name', 'Unknown')
            department = getattr(prof, 'department', 'Unknown')
            rating = getattr(prof, 'rating', 'N/A')
            difficulty = getattr(prof, 'difficulty', 'N/A')
            num_ratings = getattr(prof, 'num_ratings', 0)
            would_take_again = getattr(prof, 'would_take_again', None)
            
            would_take_again_str = (
                f"{round(would_take_again, 1)}%"
                if would_take_again is not None
                else "N/A"
            )
            
            chunk = (
                f"Professor {name} teaches in the {department} department "
                f"at Texas A&M University. "
                f"RateMyProfessor rating: {rating} out of 5.0. "
                f"Difficulty: {difficulty} out of 5.0. "
                f"Would take again: {would_take_again_str}. "
                f"Based on {num_ratings} student ratings."
            )
            chunks.append(chunk)
            
        except Exception:
            skipped += 1
            continue
    
    print(f"Created {len(chunks)} professor chunks ({skipped} skipped - no ratings)")
    return chunks


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Fetch all TAMU professors from RateMyProfessor
    professors = fetch_all_tamu_professors()
    
    # Step 2: Convert to text chunks
    chunks = convert_to_text_chunks(professors)
    
    # Step 3: Save to processed folder
    output_path = os.path.join(base_dir, 'data', 'processed', 'professor_chunks.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n')
    
    print(f"\nSaved {len(chunks)} chunks to {output_path}")
    
    # Show first 3 as a sample
    print("\n--- SAMPLE OUTPUT (first 3) ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(chunk)


if __name__ == "__main__":
    main()
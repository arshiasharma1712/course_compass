'''
This file reads the big CSV grade distribution data, cleans up the data, and converts each row into a human-readable sentence like
"In Fall 2023, Professor Smith taught CSCE-221 with 45 students. Grade distribution: A: 20, B: 15, C: 8, D: 1, F: 1. 
Average GPA: 3.2."

'''
# We need two libraries:
# - pandas: reads CSV files into a table we can work with
# - os: lets us work with file paths and folders
import pandas as pd
import os

def load_grade_data(filepath):
    """
    This function reads the raw CSV and cleans it up.
    A 'function' is a reusable block of code - we define it once
    and can call it whenever we need it.
    """
    
    # pd.read_csv() reads the CSV file into a DataFrame.
    # A DataFrame is like a spreadsheet in Python - rows and columns.
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # .shape gives us (number of rows, number of columns)
    # This is just so we can see what we loaded
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Print the column names so we can confirm they match what we saw
    print(f"Columns: {list(df.columns)}")
    
    return df


def clean_grade_data(df):
    """
    Raw data is messy. This function cleans it up.
    """
    
    # Drop rows where critical columns are empty (NaN = Not a Number, 
    # which is how pandas represents missing values)
    df = df.dropna(subset=['Course_Info', 'Instructor', 'GPA'])
    
    # Fill any remaining missing grade counts with 0
    # (if a grade count is missing, it means 0 students got that grade)
    grade_columns = ['A', 'B', 'C', 'D', 'F']
    df[grade_columns] = df[grade_columns].fillna(0)
    
    # Convert grade columns to integers (whole numbers, not decimals)
    # They might have been read as floats (3.0 instead of 3)
    for col in grade_columns:
        df[col] = df[col].astype(int)
    
    print(f"After cleaning: {df.shape[0]} rows remain")
    return df


def convert_to_text_chunks(df):
    """
    This is the most important function.
    It converts each row of data into a natural language sentence.
    ChromaDB stores and searches TEXT, not spreadsheet rows.
    So we need to turn numbers into words.
    """
    
    chunks = []  # This will hold all our text sentences
    
    # iterrows() loops through every row in the DataFrame
    # 'index' is the row number, 'row' is the actual data
    for index, row in df.iterrows():
        
        # Build a readable sentence from the row's data
        # The f"..." syntax is called an f-string - it lets us
        # insert variable values directly into a string
        chunk = (
            f"Course: {row['Course_Info']}. "
            f"Professor: {row['Instructor']}. "
            f"Semester: {row['Term']} {row['Year']}. "
            f"College: {row['College']}. "
            f"Grade distribution - A: {row['A']}, B: {row['B']}, "
            f"C: {row['C']}, D: {row['D']}, F: {row['F']}. "
            f"Total students: {row['Total_Completed']}. "
            f"Average GPA: {row['GPA']}."
        )
        
        chunks.append(chunk)  # Add this sentence to our list
    
    print(f"Created {len(chunks)} text chunks")
    return chunks


def main():
    """
    main() is the entry point - the function that runs first
    when we execute this script.
    """
    
    # Build the path to our CSV file
    # os.path.join handles Windows/Mac path differences automatically
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, 'data', 'raw', 'tamu_grade_reports.csv')
    
    # Run our three functions in order
    df = load_grade_data(filepath)
    df = clean_grade_data(df)
    chunks = convert_to_text_chunks(df)
    
    # Print the first 3 chunks so we can see what they look like
    print("\n--- SAMPLE OUTPUT (first 3 chunks) ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(chunk)


# This line means: only run main() if we execute THIS file directly.
# If another file imports this one, main() won't run automatically.
if __name__ == "__main__":
    main()

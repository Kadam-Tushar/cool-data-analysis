# Notebook Cell 1: Install and Import Dependencies
# Run this cell first to install required packages
!pip install google-generativeai pandas numpy

# Notebook Cell 2: Import Libraries
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from typing import List, Optional, Dict
import time

# Notebook Cell 3: Setup Gemini API
# Replace 'your_api_key_here' with your actual Google Gemini API key
api_key = "your_api_key_here"  # You can also set this as an environment variable
genai.configure(api_key=api_key)

# Notebook Cell 4: Load Data
# Load the CSV file
df = pd.read_csv("./death_row_inmates_Texas.csv")
print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Check if Execution column exists
if 'Execution' in df.columns:
    print(f"Execution column found with {len(df['Execution'].unique()) unique values")
else:
    print("Warning: Execution column not found!")

# Display sample of Last Statement column
print("\nSample 'Last Statement' entries:")
print(df['Last Statement'].head())

# Notebook Cell 5: Define Enhanced Embedding Functions
def get_gemini_embedding(text: str, task_type: str = "retrieval_document", model_name: str = "models/embedding-001"):
    """Get embedding for a single text using Google Gemini with different task types."""
    try:
        if not text or text.strip() == "" or text.lower() == "decline":
            return None
            
        cleaned_text = text.strip()
        
        # Map task types to Gemini task types
        gemini_task_map = {
            "retrieval_document": "retrieval_document",
            "semantic_similarity": "retrieval_query",  # Gemini uses retrieval_query for similarity
            "classification": "retrieval_document",    # Use document retrieval for classification
            "clustering": "retrieval_document"         # Use document retrieval for clustering
        }
        
        gemini_task = gemini_task_map.get(task_type, "retrieval_document")
        
        embedding = genai.embed_content(
            model=model_name,
            content=cleaned_text,
            task_type=gemini_task
        )
        return embedding['embedding']
        
    except Exception as e:
        print(f"Error getting embedding for text: {e}")
        return None

def get_embeddings_for_column(df: pd.DataFrame, column_name: str, task_types: List[str] = None, batch_size: int = 10):
    """Get embeddings for all values in a specified column for different task types."""
    if task_types is None:
        task_types = ["retrieval_document", "semantic_similarity", "classification", "clustering"]
    
    embeddings_dict = {task_type: [] for task_type in task_types}
    total_rows = len(df)
    
    print(f"Processing {total_rows} rows for column '{column_name}' with {len(task_types)} task types...")
    
    for i, text in enumerate(df[column_name]):
        if i % batch_size == 0:
            print(f"Processing row {i+1}/{total_rows}")
            time.sleep(0.1)  # Small delay to avoid rate limiting
            
        # Generate embeddings for each task type
        for task_type in task_types:
            embedding = get_gemini_embedding(text, task_type)
            embeddings_dict[task_type].append(embedding)
        
        if (i + 1) % 50 == 0:
            print(f"Completed {i+1}/{total_rows} embeddings for all task types")
    
    print(f"Completed all {total_rows} embeddings for all task types!")
    return embeddings_dict

def save_comprehensive_embeddings(embeddings_dict: Dict[str, List[Optional[List[float]]]], 
                                 df: pd.DataFrame, 
                                 column_name: str,
                                 output_file: str = "gemini_embeddings_comprehensive.csv"):
    """Save comprehensive CSV file with all task types and Execution column."""
    # Get valid embeddings for the first task type to determine dimension
    first_task = list(embeddings_dict.keys())[0]
    valid_embeddings = [emb for emb in embeddings_dict[first_task] if emb is not None]
    
    if not valid_embeddings:
        print("No valid embeddings to save!")
        return None
    
    embedding_dim = len(valid_embeddings[0])
    result_data = []
    
    for i, text in enumerate(df[column_name]):
        # Skip if no valid embeddings for any task type
        if not any(embeddings_dict[task][i] is not None for task in embeddings_dict.keys()):
            continue
            
        row_data = {}
        
        # Add Execution column
        if 'Execution' in df.columns:
            row_data['Execution'] = df.iloc[i]['Execution']
        
        # Add original text
        row_data['original_text'] = text
        
        # Add embeddings for each task type
        for task_type, embeddings in embeddings_dict.items():
            if embeddings[i] is not None:
                for j, value in enumerate(embeddings[i]):
                    row_data[f"{task_type}_dim_{j}"] = value
            else:
                # Fill with NaN for failed embeddings
                for j in range(embedding_dim):
                    row_data[f"{task_type}_dim_{j}"] = np.nan
        
        result_data.append(row_data)
    
    # Create DataFrame and save
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(output_file, index=False)
    
    print(f"Saved {len(result_data)} rows with embeddings to {output_file}")
    print(f"Embedding dimension per task type: {embedding_dim}")
    print(f"Task types included: {list(embeddings_dict.keys())}")
    
    return result_df

def save_task_specific_files(embeddings_dict: Dict[str, List[Optional[List[float]]]], 
                            df: pd.DataFrame, 
                            column_name: str):
    """Save separate CSV files for each task type with Execution column."""
    for task_type, embeddings in embeddings_dict.items():
        # Filter valid embeddings
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        
        if not valid_indices:
            print(f"No valid embeddings for task type: {task_type}")
            continue
        
        # Create DataFrame for this task type
        task_data = []
        
        for idx in valid_indices:
            row_data = {}
            
            # Add Execution column
            if 'Execution' in df.columns:
                row_data['Execution'] = df.iloc[idx]['Execution']
            
            # Add original text
            row_data['original_text'] = df.iloc[idx][column_name]
            
            # Add embeddings
            embedding = embeddings[idx]
            for j, value in enumerate(embedding):
                row_data[f"dim_{j}"] = value
            
            task_data.append(row_data)
        
        # Create DataFrame and save
        task_df = pd.DataFrame(task_data)
        filename = f"gemini_embeddings_{task_type}.csv"
        task_df.to_csv(filename, index=False)
        
        print(f"Saved {len(task_data)} embeddings for {task_type} to {filename}")
        print(f"  - Embedding dimension: {len(embeddings[valid_indices[0]])}")
        print(f"  - Valid embeddings: {len(valid_indices)}")

# Notebook Cell 6: Generate Embeddings for Multiple Task Types
# Define task types
task_types = ["retrieval_document", "semantic_similarity", "classification", "clustering"]

# Generate embeddings for all task types
print("Generating embeddings for multiple task types...")
embeddings_dict = get_embeddings_for_column(df, 'Last Statement', task_types)

# Notebook Cell 7: Analyze Embeddings by Task Type
# Count valid embeddings for each task type
print("\nEmbedding Statistics by Task Type:")
for task_type, embeddings in embeddings_dict.items():
    valid_count = sum(1 for emb in embeddings if emb is not None)
    print(f"- {task_type}: {valid_count} valid out of {len(embeddings)} total")

# Notebook Cell 8: Save Comprehensive Embeddings File
# Save comprehensive embeddings file with all task types and Execution column
print("\nSaving comprehensive embeddings file...")
comprehensive_df = save_comprehensive_embeddings(embeddings_dict, df, 'Last Statement', "gemini_embeddings_comprehensive.csv")

# Notebook Cell 9: Save Task-Specific Files
# Save separate CSV files for each task type
print("\nSaving task-specific embedding files...")
save_task_specific_files(embeddings_dict, df, 'Last Statement')

# Notebook Cell 10: Display Final Results
# Display final statistics
print("\nFinal Summary:")
print(f"- Total rows processed: {len(df)}")
print(f"- Task types processed: {len(task_types)}")
print(f"- Files created:")
print(f"  * gemini_embeddings_comprehensive.csv (all task types)")
for task_type in task_types:
    print(f"  * gemini_embeddings_{task_type}.csv")

# Show sample of comprehensive data
if comprehensive_df is not None and len(comprehensive_df) > 0:
    print(f"\nSample comprehensive data structure:")
    print(f"Columns: {list(comprehensive_df.columns)}")
    print(f"First row sample (first 5 columns):")
    print(comprehensive_df.iloc[0, :5].tolist())

# Notebook Cell 11: Optional - Quick Embedding Test
# Test with a single statement to verify everything works
test_text = "I love you all and I'm sorry for the pain I caused."
print(f"\nTesting embeddings for: '{test_text}'")

for task_type in task_types:
    test_embedding = get_gemini_embedding(test_text, task_type)
    if test_embedding:
        print(f"Success! {task_type}: Dimension: {len(test_embedding)}")
    else:
        print(f"Failed! {task_type}")

# Notebook Cell 12: Data Analysis Examples
# Analyze the generated embeddings
if comprehensive_df is not None and len(comprehensive_df) > 0:
    print(f"\nData Analysis Examples:")
    print(f"1. Total rows with embeddings: {len(comprehensive_df)}")
    
    # Count non-NaN values for each task type
    for task_type in task_types:
        task_cols = [col for col in comprehensive_df.columns if col.startswith(f"{task_type}_dim_")]
        if task_cols:
            non_nan_count = comprehensive_df[task_cols].notna().sum().sum()
            print(f"2. {task_type}: {non_nan_count} non-NaN embedding values")
    
    # Show Execution column statistics
    if 'Execution' in comprehensive_df.columns:
        print(f"3. Execution numbers range: {comprehensive_df['Execution'].min()} to {comprehensive_df['Execution'].max()}")
        print(f"4. Unique execution numbers: {comprehensive_df['Execution'].nunique()}")

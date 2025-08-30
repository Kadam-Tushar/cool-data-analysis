import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from typing import List, Optional, Dict
import time

def setup_gemini_api(api_key: str):
    """
    Setup Google Gemini API with the provided API key.
    
    Args:
        api_key (str): Google Gemini API key
    """
    genai.configure(api_key=api_key)
    
def get_gemini_embedding(text: str, task_type: str = "retrieval_document", model_name: str = "models/embedding-001") -> Optional[List[float]]:
    """
    Get embedding for a single text using Google Gemini with different task types.
    
    Args:
        text (str): Text to embed
        task_type (str): Type of task (retrieval_document, semantic_similarity, classification, clustering)
        model_name (str): Gemini model name for embeddings
        
    Returns:
        Optional[List[float]]: Embedding vector or None if failed
    """
    try:
        if not text or text.strip() == "" or text.lower() == "decline":
            return None
            
        # Clean the text
        cleaned_text = text.strip()
        
        # Map task types to Gemini task types
        gemini_task_map = {
            "retrieval_document": "retrieval_document",
            "semantic_similarity": "retrieval_query",  # Gemini uses retrieval_query for similarity
            "classification": "retrieval_document",    # Use document retrieval for classification
            "clustering": "retrieval_document"         # Use document retrieval for clustering
        }
        
        gemini_task = gemini_task_map.get(task_type, "retrieval_document")
        
        # Get embedding
        embedding = genai.embed_content(
            model=model_name,
            content=cleaned_text,
            task_type=gemini_task
        )
        
        return embedding['embedding']
        
    except Exception as e:
        print(f"Error getting embedding for text: {e}")
        return None

def get_embeddings_for_column(df: pd.DataFrame, column_name: str, task_types: List[str] = None, batch_size: int = 10) -> Dict[str, List[Optional[List[float]]]]:
    """
    Get embeddings for all values in a specified column for different task types.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        column_name (str): Name of the column to embed
        task_types (List[str]): List of task types to generate embeddings for
        batch_size (int): Number of embeddings to process before pausing
        
    Returns:
        Dict[str, List[Optional[List[float]]]]: Dictionary with task types as keys and lists of embeddings as values
    """
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
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Completed {i+1}/{total_rows} embeddings for all task types")
    
    print(f"Completed all {total_rows} embeddings for all task types!")
    return embeddings_dict

def save_embeddings_to_csv(embeddings_dict: Dict[str, List[Optional[List[float]]]], 
                          df: pd.DataFrame, 
                          column_name: str,
                          output_file: str = "gemini_embeddings.csv"):
    """
    Save embeddings to a CSV file with Execution column and different task types.
    
    Args:
        embeddings_dict (Dict[str, List[Optional[List[float]]]]): Dictionary of embeddings by task type
        df (pd.DataFrame): Original DataFrame to get Execution column
        column_name (str): Name of the column that was embedded
        output_file (str): Output CSV filename
    """
    # Get valid embeddings for the first task type to determine dimension
    first_task = list(embeddings_dict.keys())[0]
    valid_embeddings = [emb for emb in embeddings_dict[first_task] if emb is not None]
    
    if not valid_embeddings:
        print("No valid embeddings to save!")
        return
    
    # Get embedding dimension
    embedding_dim = len(valid_embeddings[0])
    
    # Create a comprehensive DataFrame
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
    
    # Create DataFrame
    result_df = pd.DataFrame(result_data)
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Saved {len(result_data)} rows with embeddings to {output_file}")
    print(f"Embedding dimension per task type: {embedding_dim}")
    print(f"Task types included: {list(embeddings_dict.keys())}")
    
    return result_df

def save_task_specific_embeddings(embeddings_dict: Dict[str, List[Optional[List[float]]]], 
                                 df: pd.DataFrame, 
                                 column_name: str):
    """
    Save separate CSV files for each task type with Execution column.
    
    Args:
        embeddings_dict (Dict[str, List[Optional[List[float]]]]): Dictionary of embeddings by task type
        df (pd.DataFrame): Original DataFrame to get Execution column
        column_name (str): Name of the column that was embedded
    """
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

def main():
    """
    Main function to process the CSV and generate embeddings.
    """
    # Check if API key is set
    api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    if not api_key:
        print("Error: GOOGLE_GEMINI_API_KEY environment variable not set!")
        print("Please set it with: export GOOGLE_GEMINI_API_KEY='your_api_key_here'")
        return
    
    # Setup API
    setup_gemini_api(api_key)
    
    # Read the CSV file
    csv_path = "./data/death_row_inmates_Texas.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Check if 'Last Statement' column exists
    if 'Last Statement' not in df.columns:
        print("Error: 'Last Statement' column not found in the CSV!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Check if 'Execution' column exists
    if 'Execution' not in df.columns:
        print("Warning: 'Execution' column not found in the CSV!")
        print("Embeddings will be saved without execution numbers.")
    
    # Display some sample data
    print("\nSample 'Last Statement' entries:")
    print(df['Last Statement'].head())
    
    # Define task types
    task_types = ["retrieval_document", "semantic_similarity", "classification", "clustering"]
    
    # Get embeddings for all task types
    print("\nGenerating embeddings for multiple task types...")
    embeddings_dict = get_embeddings_for_column(df, 'Last Statement', task_types)
    
    # Count valid embeddings for each task type
    print("\nEmbedding Statistics by Task Type:")
    for task_type, embeddings in embeddings_dict.items():
        valid_count = sum(1 for emb in embeddings if emb is not None)
        print(f"- {task_type}: {valid_count} valid out of {len(embeddings)} total")
    
    # Save comprehensive embeddings file
    print("\nSaving comprehensive embeddings file...")
    comprehensive_df = save_embeddings_to_csv(embeddings_dict, df, 'Last Statement', "gemini_embeddings_comprehensive.csv")
    
    # Save task-specific files
    print("\nSaving task-specific embedding files...")
    save_task_specific_embeddings(embeddings_dict, df, 'Last Statement')
    
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

if __name__ == "__main__":
    main()

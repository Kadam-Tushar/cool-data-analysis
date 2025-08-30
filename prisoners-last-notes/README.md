# Google Gemini Embeddings for Prisoner Last Statements

This project generates Google Gemini embeddings for the 'Last Statement' column from the Texas death row inmates dataset, with support for multiple task types and inclusion of the Execution column.

## Files

- `embeddings.py` - Standalone Python script for generating embeddings
- `gemini_embeddings_notebook.py` - Notebook-ready code cells
- `requirements.txt` - Required Python packages
- `death_row_inmates_Texas.csv` - Dataset containing prisoner last statements

## Features

### 🎯 **Multiple Task Types**
Generates embeddings optimized for different NLP tasks:
- **`retrieval_document`** - For document retrieval and general text analysis
- **`semantic_similarity`** - For finding similar statements and semantic search
- **`classification`** - For categorizing statements by themes or emotions
- **`clustering`** - For grouping similar statements together

### 📊 **Execution Column Integration**
- Automatically includes the `Execution` column from the original dataset
- Links embeddings back to specific execution numbers for traceability
- Maintains data integrity and relationships

### 💾 **Multiple Output Formats**
- **Comprehensive CSV**: All task types in one file with Execution numbers
- **Task-specific CSVs**: Separate files for each embedding type
- **Original text preservation**: Keeps the source statements for reference

## Setup

### 1. Get Google Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Key

**Option A: Environment Variable (Recommended)**
```bash
export GOOGLE_GEMINI_API_KEY='your_api_key_here'
```

**Option B: Direct in Code**
Edit the script and replace `"your_api_key_here"` with your actual API key.

## Usage

### Option 1: Standalone Script

Run the standalone script:
```bash
python embeddings.py
```

### Option 2: Jupyter Notebook

1. Open your existing notebook (`last_words_eda.ipynb`)
2. Copy the code cells from `gemini_embeddings_notebook.py`
3. Replace `"your_api_key_here"` with your actual API key
4. Run the cells sequentially

### Option 3: Import Functions

```python
from embeddings import get_gemini_embedding, get_embeddings_for_column

# Get single embedding for specific task
embedding = get_gemini_embedding("Some text here", task_type="semantic_similarity")

# Get embeddings for entire column with multiple task types
task_types = ["retrieval_document", "semantic_similarity", "classification", "clustering"]
embeddings_dict = get_embeddings_for_column(df, 'Last Statement', task_types)
```

## Enhanced Features

- **Batch Processing**: Processes embeddings in batches to avoid rate limiting
- **Error Handling**: Gracefully handles failed embeddings and empty text
- **Progress Tracking**: Shows progress during processing for all task types
- **Data Validation**: Filters out "decline" statements and empty text
- **Multiple CSV Export**: Saves embeddings in various formats for different use cases
- **Statistics**: Provides detailed statistics about the embedding process for each task type

## Output Files

The script generates multiple output files:

### 1. Comprehensive File (`gemini_embeddings_comprehensive.csv`)
Contains all task types in one file with columns:
- `Execution` - Execution number from original dataset
- `original_text` - The original last statement
- `{task_type}_dim_{i}` - Embedding dimensions for each task type

### 2. Task-Specific Files
- `gemini_embeddings_retrieval_document.csv`
- `gemini_embeddings_semantic_similarity.csv`
- `gemini_embeddings_classification.csv`
- `gemini_embeddings_clustering.csv`

Each task-specific file contains:
- `Execution` - Execution number
- `original_text` - Original statement
- `dim_{i}` - Embedding dimensions

## Embedding Models

Uses Google Gemini's `models/embedding-001` model with task-specific optimizations:

- **Document Retrieval**: Optimized for finding relevant documents
- **Semantic Similarity**: Optimized for finding similar text passages
- **Classification**: Optimized for categorizing text content
- **Clustering**: Optimized for grouping similar content

## Data Handling

- **Filters out**: Empty statements, "decline" responses
- **Processes**: All other valid last statements
- **Handles**: Special characters, long text, various languages
- **Preserves**: Execution numbers and original text for traceability

## Rate Limiting

The script includes small delays between API calls to respect rate limits. Adjust the `batch_size` and `time.sleep()` values if needed.

## Example Output Structure

```
Execution,original_text,retrieval_document_dim_0,retrieval_document_dim_1,...,semantic_similarity_dim_0,...
570,"Yes sir. I would like to thank God...",0.123,0.456,...,0.789,0.012,...
569,"Thank you thank you where's the family...",0.234,0.567,...,0.890,0.123,...
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your API key is correct and has proper permissions
2. **Rate Limiting**: Reduce batch size or increase delays between calls
3. **Memory Issues**: Process smaller batches for large datasets
4. **Network Errors**: Check internet connection and API availability

### Error Messages

- `GOOGLE_GEMINI_API_KEY environment variable not set`: Set your API key
- `Error reading CSV`: Check file path and format
- `Column not found`: Verify column name matches exactly
- `Error getting embedding`: Check API key and network connection

## Example Output

```
Loaded 569 rows
Columns: ['Execution', 'Last Name', 'First Name', 'TDCJNumber', 'Age', 'Date', 'Race', 'County', 'Last Statement']
Execution column found with 569 unique values

Sample 'Last Statement' entries:
0    decline
1    Yes sir. I would like to thank God, my dad, my Lord Jesus savior...
2    Thank you thank you where's the family, ok I would like to say sorry...

Generating embeddings for multiple task types...
Processing 569 rows for column 'Last Statement' with 4 task types...
Processing row 1/569
Processing row 11/569
...
Completed all 569 embeddings for all task types!

Embedding Statistics by Task Type:
- retrieval_document: 450 valid out of 569 total
- semantic_similarity: 450 valid out of 569 total
- classification: 450 valid out of 569 total
- clustering: 450 valid out of 569 total

Saving comprehensive embeddings file...
Saved 450 rows with embeddings to gemini_embeddings_comprehensive.csv
Embedding dimension per task type: 768
Task types included: ['retrieval_document', 'semantic_similarity', 'classification', 'clustering']

Saving task-specific embedding files...
Saved 450 embeddings for retrieval_document to gemini_embeddings_retrieval_document.csv
  - Embedding dimension: 768
  - Valid embeddings: 450
...
```

## Next Steps

After generating embeddings, you can:

1. **Analyze similarities** between statements using semantic_similarity embeddings
2. **Cluster statements** by themes using clustering embeddings
3. **Classify statements** by emotion or content using classification embeddings
4. **Build search functionality** for finding similar statements
5. **Train machine learning models** for various NLP tasks
6. **Perform cross-task analysis** comparing embeddings across different task types

## Use Cases by Task Type

- **`retrieval_document`**: General text analysis, document search, content recommendation
- **`semantic_similarity`**: Finding similar statements, semantic search, duplicate detection
- **`classification`**: Emotion analysis, theme categorization, content tagging
- **`clustering`**: Grouping similar statements, topic modeling, pattern discovery

## License

This project is for educational and research purposes. Please ensure compliance with Google's API terms of service.

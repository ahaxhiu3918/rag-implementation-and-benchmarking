"""
Here are all the functions for processing data

"""
import json
from sentence_transformers import SentenceTransformer


def extract_code_cells_from_notebook(notebook_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Notebook file not found: {notebook_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Notebook file is not valid JSON: {notebook_path}")

    code_blocks = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            code = ''.join(cell.get('source', []))
            code_blocks.append(code)
    return code_blocks

def process_user_query(query: str):
    """
    Process user query and convert to embedding for vector search.
    
    """
    
    # Loading embedding model 
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Preprocess query
    cleaned_query = query.lower().strip()
    
    # Convert query to embedding
    query_embedding = model.encode([cleaned_query])

    return query_embedding[0]


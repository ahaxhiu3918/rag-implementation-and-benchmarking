import json
import os
import time
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from transformers import pipeline

#=======
# Model Loading
#=======

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch

model_id = "bigcode/starcoder2-3b"

# 1. Define the Quantization Config separately
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16 # Recommended for better performance
)

# 2. Load the model and tokenizer explicitly
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Create the pipeline WITHOUT the loading flags
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

#======
# Pipeline & wrappeur
#======

def extract_code_cells_from_notebook(input_data):
    """
    Args:
        input_data (str): Either a path to a .ipynb file OR a raw code string.

    Returns:
        list: A list of code strings.
    """
    # 1. Check if the input is a path to a notebook file
    if isinstance(input_data, str) and input_data.endswith('.ipynb') and os.path.exists(input_data):
        try:
            with open(input_data, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            code_blocks = []
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    # Notebook source can be a list of lines or a single string
                    code = ''.join(source) if isinstance(source, list) else source
                    if code.strip():  # Only add non-empty cells
                        code_blocks.append(code)
            return code_blocks
            
        except (json.JSONDecodeError, KeyError):
            # If it's a .ipynb but malformed, treat it as a raw string or raise error
            raise ValueError(f"File at {input_data} is not a valid Jupyter Notebook.")
    
    # 2. If it's not a notebook path, treat the entire string as a single code chunk
    else:
        # Wrap in a list to keep the return type consistent
        return [input_data] if input_data.strip() else []


def load_and_chunk_documents():
    """
    Load sample notebooks example classfication and chunk them for better retrieval.
    
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    policy_documents = [

    {
        "desc": "loading_data",
        "code": """
import pandas as pd
df = pd.read_csv('data.csv')
"""
    },
    {
        "desc": "loading_data",
        "code": """
import pandas as pd
import sqlalchemy
engine = sqlalchemy.create_engine('sqlite:///data.db')
df = pd.read_sql('SELECT * FROM table', engine)
"""
    },
    {
        "desc": "loading_data",
        "code": """
import pandas as pd
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
"""
    },
    {
        "desc": "preprocessing",
        "code": """
df.fillna(df.mean(), inplace=True)
X = df.drop('target', axis=1)
y = df['target']
"""
    },
    {
        "desc": "preprocessing",
        "code": """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
    },
    {
        "desc": "preprocessing",
        "code": """
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""
    },
    {
        "desc": "preprocessing",
        "code": """
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])
"""
    },
    {
        "desc": "preprocessing",
        "code": """
X = pd.get_dummies(df.drop('target', axis=1), drop_first=True)
y = df['target']
"""
    },
    {
        "desc": "preprocessing",
        "code": """
df['log_value'] = df['value'].apply(lambda x: np.log(x + 1e-9))
"""
    },
    {
        "desc": "loading_model",
        "code": """
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
"""
    },
    {
        "desc": "loading_model",
        "code": """
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
"""
    },
    {
        "desc": "loading_model",
        "code": """
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
"""
    },
    {
        "desc": "fit",
        "code": """
model.fit(X_train, y_train)
"""
    },
    {
        "desc": "fit",
        "code": """
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'Cross-validated accuracy: {scores.mean():.2f}')
"""
    },
    {
        "desc": "fit",
        "code": """
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
model = grid.best_estimator_
"""
    },
    {
        "desc": "results",
        "code": """
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
"""
    },
    {
        "desc": "results",
        "code": """
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()
"""
    },
    {
        "desc": "results",
        "code": """
importances = model.feature_importances_
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importance.sort_values('importance', ascending=False, inplace=True)
print(feature_importance.head(10))
"""
    },
    {
        "desc": "exploratory_analysis",
        "code": """
print(df.shape)
print(df.dtypes)
print(df.describe())
"""
    },
    {
        "desc": "exploratory_analysis",
        "code": """
print(df.isnull().sum())
print(y.value_counts())
"""
    },
    {
        "desc": "inference",
        "code": """
y_pred = model.predict(X_test)
print(y_pred[:10])
"""
    },
    {
        "desc": "inference",
        "code": """
sample = X_test.iloc[0:1]
prediction = model.predict(sample)
probability = model.predict_proba(sample)
print(f'Predicted class: {prediction[0]}, Probability: {probability.max():.2f}')
"""
    }
]

    print(f"Loaded notebook code examples classification successful")
    
    # Configure text splitter 
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Chunk all documents
    all_chunks = []
    for doc in policy_documents:
        chunks = text_splitter.split_text(doc["code"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "desc": doc["desc"],
                "code": chunk,
            })
    
    return all_chunks


def setup_vector_database(chunks):
    """
    Set up ChromaDB vector database and store document chunks.
    
    """
    import chromadb
    """ 
    
    try:
        client.delete_collection(name="notebook_examples_policy")
    except:
        pass
    """
    # Initialize ChromaDB client
    client = chromadb.Client()
    
    # Create collection or get it
    
    collection = client.get_or_create_collection(name="notebook_examples_policy")#  collection name with underscores and no spaces
    
    
    # Prepare data for storage
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    codes = [chunk["code"] for chunk in chunks]
    descriptions = [{"desc": chunk["desc"]} for chunk in chunks]
    
    # Adding documents to collection & embedding
    collection.add(
        ids=ids,
        documents=codes,
        metadatas=descriptions
    )
        
    print("Vector Database Created")
    return collection

def process_user_query(query: str):
    """
    Process user query and convert to embedding for vector search.
    
    """
    from sentence_transformers import SentenceTransformer
    
    # Loading embedding model 
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Preprocess query
    cleaned_query = str(query).lower().strip()
    
    # Convert query to embedding
    query_embedding = model.encode(cleaned_query)

    return query_embedding


def search_vector_database(collection, query_embedding, top_k: int = 3):
    """
    Search vector database for relevant document chunks.
    """
    
    # Performing vector search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    # Process results
    search_results = []
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        document = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        
        similarity = 1 - distance
        search_results.append({
            'id': doc_id,
            'content': document,
            'metadata': metadata,
            'similarity': similarity
        })
    
    return search_results


def augment_prompt_with_context(query: str, search_results: List[Dict]) -> str:
    """
    Build augmented prompt with retrieved context for LLM.
    """
    # Assemble context from search results (limit to 3 unique examples)
    context_parts = []
    seen_labels = set()
    for i, result in enumerate(search_results, 1):
        label = result['metadata'].get('desc', 'unknown')
        if label not in seen_labels:
            seen_labels.add(label)
            # Fix: Use proper string formatting for the dictionary-like structure
            context_parts.append(f"Source {i}: {{'desc': '{label}'}}\n{result['content']}")
            if len(seen_labels) >= 3:  # Only include 3 unique examples
                break

    context = "\n\n".join(context_parts)

    # Build augmented prompt with explicit instructions
    augmented_prompt = f"""
You are an assistant. Your task is to classify the following notebook code chunk **only** based on the provided examples.
Classify the given notebook chunk using **only the most relevant label** as a single string.
Do not repeat the context or provide explanations. Do not list multiple sources.

Context (examples):
{context}

Given notebook chunk:
{query}

Answer (single string):
"""

    return augmented_prompt


def generate_response(augmented_prompt):
    # Generate response with the model
    response = generator(
        augmented_prompt,
        max_new_tokens=100,
        temperature=None,
        do_sample=False
    )

    # Extract the generated text
    generated_text = response[0]["generated_text"]

    # Check for structured output like {'desc': 'exploratory_analysis'}
    import re
    match = re.search(r"\{'desc':\s*'([^']+)'\}", generated_text)
    if match:
        return match.group(1)

    # Check for lines that look like a label (e.g., exploratory_analysis, results, etc.)
    possible_labels = ['loading_data', 'preprocessing', 'loading_model', 'fit', 'results', 'exploratory_analysis', 'inference']
    lines = generated_text.split('\n')
    for line in reversed(lines):
        line = line.strip().strip('\'"[]{} ')
        if line in possible_labels:
            return line

    # Check for any plausible label in the text
    for label in possible_labels:
        if f"'{label}'" in generated_text or f'"{label}"' in generated_text or f"{label}:" in generated_text:
            return label

    # Fallback if no label is found
    return "unknown"


def rag_pipeline(query01):
    """
    Running the complete RAG pipeline from start to finish.
    Here the query parameter is the notebook we put in entry, for it to be classified by a RAG pepiline using starcoder2-3b.
    The query01 input is a raw string to where the notebook in json is located, for it to be preprocessed and input into the RAG pepeline
    """
    #step0

    query = extract_code_cells_from_notebook(query01)

    # Step 1: Load and chunk documents
    chunks = load_and_chunk_documents()
    
    # Step 2: Setup vector database
    collection = setup_vector_database(chunks)
    print(f"DEBUG - collection = {collection}")  
    print(f"DEBUG - type = {type(collection)}")
    # Step 3: Process user query
    query_embedding = process_user_query(query)
    print(f"DEBUG - embedding shape: {query_embedding.shape}")
    # Step 4: Search vector database
    search_results = search_vector_database(collection, query_embedding)
    
    # Step 5: Augment prompt with context
    augmented_prompt = augment_prompt_with_context(query, search_results)
    
    # Step 6: Generate response
    response = generate_response(augmented_prompt)
    
    # Display final result
    print("FINAL RESULT\n")
    
    
    return response

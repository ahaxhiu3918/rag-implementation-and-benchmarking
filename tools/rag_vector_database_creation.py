from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

def load_and_chunk_documents():
    """
    Load sample notebooks example classfication and chunk them for better retrieval.
    
    """
    
    policy_documents = [
        {
            "desc": "Classic ML fit (should detect Model Training)",
        "code": """ 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)"""
        },
        {
             "desc": "Data loading (should detect Data Loading)",
        "code": """
import pandas as pd
df = pd.read_csv('data.csv')
"""
        },
        {
             "desc": "Model evaluation (should detect Model Evaluation)",
        "code": """
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
"""
        },
        {
            "desc": "Deep learning training (should detect Model Training, Deep Learning)",
        "code": """
import tensorflow as tf
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5)
"""
        },
        {
            "desc": "Empty code (should return nothing or fallback)",
        "code": ""
        },
        {
            "desc": "Non-ML code (should not hallucinate ML steps)",
        "code": """
print("Hello, world!")
a = 5 + 3
"""
        },
        {
            "desc": "Multiple ML steps in one cell (should detect all present steps)",
        "code": """
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
"""
        },
        {
            "desc": "Unusual ML library (should fallback or try to detect)",
        "code": """
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
"""
        },
        {
            "desc": "Malformed code (should not crash, should fallback)",
        "code": """
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
"""
        },
        {
            "desc": "Comments only (should fallback)",
        "code": """
# This cell only contains comments
# No executable code here"""
        },
        {
            "desc": "Irrelevant imports (should not hallucinate ML steps)",
        "code": """
import os
import sys
print(os.getcwd())
"""
        },
        {
            "desc": "Ambiguous variable names (should not guess)",
        "code": """
foo = bar(X, y)
baz = foo.predict(X)
"""

        },
        {
            "desc": "Only function definitions (should fallback)",
        "code": """
def my_function(x):
    return x * 2
"""

        },
        {
            "desc": "ML and non-ML mixed (should only detect ML parts)",
        "code": """
import pandas as pd
df = pd.read_csv('data.csv')
print("Loaded data")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(df['X'], df['y'])
print("Done")
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
 
    # Initialize ChromaDB client
    client = chromadb.Client()
    
    # Create collection 
    
    collection = client.create_collection(
        name="notebook examples policy",  #  collection name
    )
    
    
    # Prepare data for storage
    ids = [f"{i}" for i in len(chunks)]
    codes = [chunk["code"] for chunk in chunks]
    descriptions = [{"desc": chunks["desc"]}]
    
    # Adding documents to collection & embedding
    collection.add(
        ids = ids,
        documents=codes,
        metadatas=descriptions
        )
        
    
    return collection


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
    for i, (collection.id, distance, collection.documents, collection.metadata) in enumerate(zip(
        results['ids'][0], 
        results['distances'][0], 
        results['documents'][0], 
        results['metadatas'][0]
    )):
        similarity = 1 - distance  # Convert distance to similarity
        search_results.append({
            'id': collection.id,
            'content': collection.documents,
            'metadata': collection.metadata,
            'similarity': similarity
        })
    
    return search_results

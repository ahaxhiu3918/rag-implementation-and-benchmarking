# RAG Pipeline with StarCoder2-3B

A **basic implementation of a Retrieval-Augmented Generation (RAG) pipeline** using the [StarCoder2-3B](https://huggingface.co/bigcode/starcoder2) model. This project provides a modular framework for experimenting with RAG, evaluating its performance, and benchmarking different implementations.

---

## 📦 Project Structure

RAG_env/                  # Project environment

│

├── RESULT/                # Contains benchmark results (Excel file)

│   └── benchmark_results.xlsx

│

├── demo/                  # Kaggle notebook for running the benchmark

│   └── kaggle_benchmark.ipynb

│

├── models/                # RAG implementations

│   ├── rag01.py           # Initial implementation

│   ├── rag02.py           # Intermediate implementation

│   └── rag03.py           # Newest implementation

│

├── rag_eval_framework/    # Framework for evaluating RAG with an LLM

│   ├── evaluator.py       # Evaluation logic

│   └── metrics.py         # Custom metrics

│

├── rag_notebooks_data/    # Datasets for the project

│   ├── dataset1.json

│   ├── dataset2.csv

│   └── ...

│

└── tools/                 # Utility functions for RAG

│    ├── rag_metrics.py     # Metrics for RAG evaluation

##  How to Use

### Prerequisites
- Python 3.11+
- Install dependencies:
  ```bash
  pip install -r requirements.txt

- Access to StarCoder2-3B (via Hugging Face Transformers or similar)

## Quick Start


### Core Files:

- Use models/rag03.py for the newest RAG implementation.
- Use tools/rag_metrics.py for metrics to evaluate RAG performance.


### Run a Benchmark:

Use the notebook dev04_evaluation.ipynb to evaluate the RAG pipeline.

### Benchmark Results:

Results are saved in RESULT/

### 🔍 Features

 - Fixed RAG Pipeline: starcoder2-3b
 - Evaluation Framework: Benchmark RAG performance with custom metrics.
 - Datasets: Pre-loaded datasets for testing and development.
 - Kaggle Integration: Ready-to-run notebook for benchmarking.

### 📊 Benchmarking

The demo/ notebook runs a small benchmark and saves results to RESULT/.
Metrics include:

 - F1_score
 - Kohen's Cappa
 - Semantic Consistency
 - Precision
 - Recall 


### 🛠️ Customization & Improvements

 - Add New Models: Place new RAG implementations in models/. (other than starcoder2-3n)
 - Extend Datasets: Add datasets to rag_notebooks_data/.
 - New Metrics: Add custom metrics in tools/rag_metrics.py.





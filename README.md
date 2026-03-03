# SimilarityTextDetector

A simple web application that measures similarity between two texts using different metrics.


## Features

- Jaccard similarity (unique word overlap)
- Cosine similarity using TF-IDF
- Extraction of common key terms
- Basic statistics (token counts, unique words, etc.)


## How It Works

The application:

1. Tokenizes both texts
2. Computes Jaccard similarity based on unique words
3. Computes cosine similarity using TF-IDF vectors
4. Extracts the most relevant common terms
5. Returns structured JSON results via a FastAPI backend

> Input:
Take two texts as input

> Output:
Scores of similarity between the two texts.


## Installation

### Packages
- fastapi
- uvicorn
- scikit-learn

### 1. Create a virtual environment (recommended)
To install in virtual environment:
````py
python -m venv <name_of_venv>
````

Activate it:
**Windows**
````py
<venv>\>Scripts\activate
````
**Linux/macOS**
````py
source <venv>/bin/activate
````

### 2. Install dependencies
````py
pip install fastapi uvicorn scikit-learn
````

## Run the application
To start the FastAPI server, run:
````py
uvicorn main:app --reload
````

Then open (local version):
````py
http://127.0.1:8000
````
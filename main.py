import math
import re
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", re.UNICODE)

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

def safe_round(text_number: float):
    try:
        if math.isnan(text_number) or math.isinf(text_number):
            return 0.0
        return float(round(text_number, 4))
    except:
        return text_number

    
### SCORES ###
def jaccard_similarity(tokens1: list[str], tokens2: list[str]) -> float:
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 and not set2:
        return 1.0  # if both sets are empty, return perfect similarity
    if not set1 or not set2:
        return 0.0  # if one set is empty, return no similarity
    #return jaccard_score(list(set1), list(set2))
    return len(set1 & set2) / len(set1 | set2)

def cosine_tfidf(text1: str, text2: str) -> float:
    vect = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)[A-Za-zÀ-ÖØ-öø-ÿ0-9']+",    # (?u) for unicode support
        ngram_range=(1, 2)                              # unigrams and bigrams (grams means sequences of tokens)
    )
    x = vect.fit_transform([text1, text2])
    return float(cosine_similarity(x[0], x[1])[0][0])   # [0][0] to get the single value from the 1x1 matrix returned by cosine_similarity

def top_common_terms(text1: str, text2: str, n: int = 10) -> list[str]:
    """
    - TF-IDF on (A, B)
    - we take the words with high weight in A AND in B (min(vA, vB))
    """
    vect = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)[A-Za-zÀ-ÖØ-öø-ÿ0-9']+",    # (?u) for unicode support
        ngram_range=(1, 1)                              # unigrams and bigrams (grams means sequences of tokens)
    )
    x = vect.fit_transform([text1, text2])  # fit_transform to learn the vocabulary and transform the texts into TF-IDF vectors
    a = x[0].toarray().ravel()  # convert the sparse matrix to a dense array and flatten it
    b = x[1].toarray().ravel()
    common_score = [min(ai, bi) for ai, bi in zip(a, b)]
    vocab = vect.get_feature_names_out()
    pairs = [(vocab[i], common_score[i]) for i in range(len(vocab)) if common_score[i] > 0]
    pairs.sort(key=lambda x:x[1], reverse=True)
    return [term for term, score in pairs[:n]]

### CLASSES ###
class ComparePayload(BaseModel):
    textA: str = Field(default="", max_length=200000)
    textB: str = Field(default="", max_length=200000)

### MAIN PART ###
app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def home():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/api/compare")
def compare(payload: ComparePayload):
    text1 = payload.textA or ""
    text2 = payload.textB or ""
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    jaccard = jaccard_similarity(tokens1, tokens2)
    cosine = cosine_tfidf(text1, text2)
    # to avoid doing the same tokenization again, we can compute the common tokens here and pass them to top_common_terms, in case text too long
    common = sorted(set(tokens1) & set(tokens2))
    common_limited = common[:1000]  # limit to 1000 common tokens to avoid performance issues
    keyterms = top_common_terms(text1, text2, n = 10)

    return {
        "jaccard": jaccard,
        "cosine_tfidf": safe_round(cosine),
        "common_tokens_sample": common_limited,
        "top_common_terms": keyterms,
        "counts": {
            "tokensA": len(tokens1),
            "tokensB": len(tokens2),
            "uniqueA": len(set(tokens1)),
            "uniqueB": len(set(tokens2)),
            "common_unique": len(set(tokens1) & set(tokens2))
        }
    }

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode embeddings
df["embeddings"] = df["plot"].apply(lambda x: model.encode(x))

def search_movies(query, top_n=5):
    """
    Returns top_n movie matches for a given query
    """
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], list(df["embeddings"]))[0]
    df["similarity"] = similarities
    results = df.sort_values("similarity", ascending=False).head(top_n)
    return results[["title", "plot", "similarity"]]

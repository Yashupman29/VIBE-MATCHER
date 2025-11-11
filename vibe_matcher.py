"""
Vibe Matcher — Mini Fashion Recommender
Usage:
    python vibe_matcher.py
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

products = [
    {"name": "Boho Dress", "desc": "Flowy, earthy tones and layered textiles for festival and sunset vibes.", "tags": ["boho", "festival", "earthy"]},
    {"name": "Energetic Bomber", "desc": "Bold colors, sharp cuts — urban energy meets street-smart tailoring.", "tags": ["urban", "energetic", "street"]},
    {"name": "Cozy Knit Sweater", "desc": "Soft wool, oversized fit — perfect for cozy coffee dates and chilly evenings.", "tags": ["cozy", "casual", "warm"]},
    {"name": "Minimalist Trench", "desc": "Clean lines, neutral palette — timeless urban-chic for work-to-evening.", "tags": ["minimalist", "urban", "chic"]},
    {"name": "Athleisure Runner", "desc": "Breathable fabrics and dynamic silhouettes for high-energy movement.", "tags": ["athleisure", "energetic", "active"]},
    {"name": "Ethereal Maxi", "desc": "Lightweight, pastel layers with delicate embroidery — dreamy, boho-luxe.", "tags": ["boho", "ethereal", "dreamy"]},
    {"name": "Street Hoodie", "desc": "Relaxed fit, logo details, engineered for everyday urban comfort.", "tags": ["street", "casual", "urban"]},
    {"name": "Tailored Blazer", "desc": "Structured shoulders and refined tailoring — boardroom-ready with a modern twist.", "tags": ["formal", "chic", "tailored"]},
]
df = pd.DataFrame(products)

def get_embedding(text, model="text-embedding-ada-002"):
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

print("Generating embeddings...")
df["embedding"] = df["desc"].apply(get_embedding)
embedding_matrix = np.vstack(df["embedding"])
print("Done.")

def top_k_matches(query, k=3, threshold=0.7):
    q_emb = get_embedding(query)
    sims = cosine_similarity([q_emb], embedding_matrix)[0]
    idx_sorted = np.argsort(-sims)
    results = []
    for i in idx_sorted[:k]:
        results.append({
            "rank": len(results)+1,
            "name": df.loc[i,"name"],
            "score": float(sims[i]),
            "tags": df.loc[i,"tags"]
        })
    if results[0]["score"] < threshold:
        results.append({"note": "Top score below threshold — fallback engaged."})
    return results

queries = ["energetic urban chic", "cozy weekend brunch", "ethereal boho festival"]
for q in queries:
    print(f"\n🧭 Query: {q}")
    results = top_k_matches(q)
    for r in results:
        print(r)

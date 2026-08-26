from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "spent 200 on groceries",
    "bought vegetables from market",
    "paid electricity bill"
]

embeddings = model.encode(sentences)

score1 = cosine_similarity([embeddings[0]], [embeddings[1]])
score2 = cosine_similarity([embeddings[0]], [embeddings[2]])
score3 = cosine_similarity([embeddings[1]], [embeddings[2]])

print(f"Groceries vs Vegetables: {score1[0][0]:.2f}")
print(f"Groceries vs Electricity: {score2[0][0]:.2f}")
print(f"Vegetables vs Electricity: {score3[0][0]:.2f}")

import os
from dotenv import load_dotenv
from groq import Groq
import chromadb

load_dotenv()

# Setup Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Setup ChromaDB with sample transactions
chroma = chromadb.Client()
collection = chroma.create_collection("finance")

collection.add(
    documents=[
        "spent 200 on groceries",
        "paid 500 electricity bill",
        "bought gym membership for 1000",
        "ordered food from swiggy for 300",
        "paid rent 8000"
    ],
    ids=["t1", "t2", "t3", "t4", "t5"]
)

# User asks a question
question = "how much did I spend on food?"

# Step 1 - Retrieve relevant transactions from ChromaDB
results = collection.query(
    query_texts=[question],
    n_results=2
)

retrieved = results['documents'][0]
context = "\n".join(retrieved)

print(f"Retrieved context:\n{context}\n")

# Step 2 - Send to Groq LLM with context
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": "You are a personal finance assistant. Answer questions based only on the transactions provided."
        },
        {
            "role": "user",
            "content": f"Transactions:\n{context}\n\nQuestion: {question}"
        }
    ]
)

print(f"SHYRA: {response.choices[0].message.content}")
import chromadb

client = chromadb.Client()
collection = client.create_collection("finance")

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

results = collection.query(
    query_texts=["food expenses"],
    n_results=2
)

print(results['documents'])

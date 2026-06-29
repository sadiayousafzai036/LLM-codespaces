import numpy as np
from tqdm.auto import tqdm
from gitsource import GithubRepositoryDataReader, chunk_documents
from embedder import Embedder
from minsearch import VectorSearch, Index

# ==========================================
# SETUP
# ==========================================
embed = Embedder()

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]
print(f"Loaded {len(documents)} documents.\n")

# ==========================================
# Q1. Embedding a query
# ==========================================
query_q1 = "How does approximate nearest neighbor search work?"
v = embed.encode(query_q1)

print(f"First value of the query vector (v[0]): {v[0]:.4f}")

# ==========================================
# Q2. Cosine similarity
# ==========================================
target_filename = "02-vector-search/lessons/07-sqlitesearch-vector.md"
target_doc = next(doc for doc in documents if doc["filename"] == target_filename)

# Embed the document content
v_doc = embed.encode(target_doc["content"])

# Calculate cosine similarity (dot product of normalized vectors)
cosine_sim = v.dot(v_doc)
print(f"Cosine similarity with {target_filename}: {cosine_sim:.4f}")

# ==========================================
# Q3. Chunking and search by hand
# ==========================================
chunks = chunk_documents(documents, size=2000, step=1000)
print(f"Created {len(chunks)} chunks.")

print("Embedding chunks in batches...")
batch_size = 50
vectors = []
for i in tqdm(range(0, len(chunks), batch_size)):
    batch = [c["content"] for c in chunks[i:i + batch_size]]
    batch_vectors = embed.encode_batch(batch)
    vectors.extend(batch_vectors)

# Stack into a 2D matrix
X = np.array(vectors)
print(f"Matrix X shape: {X.shape}")

scores = X.dot(v)
best_idx = np.argmax(scores)
best_chunk = chunks[best_idx]

print(f"Highest scoring chunk filename: {best_chunk['filename']}")

# ==========================================
# Q4. Vector search with minsearch
# ==========================================
vindex = VectorSearch()
vindex.fit(X, chunks)

query_q4 = "What metric do we use to evaluate a search engine?"
v_q4 = embed.encode(query_q4)

# Search and get the top 1 result
results_q4 = vindex.search(v_q4, num_results=1)
print(f"First result filename for Q4: {results_q4[0]['filename']}")

# ==========================================
# Q5. Text search vs vector search
# ==========================================
tindex = Index(text_fields=["content"])
tindex.fit(chunks)

query_q5 = "How do I store vectors in PostgreSQL?"
v_q5 = embed.encode(query_q5)

# Get top 5 from both methods
vec_results_q5 = vindex.search(v_q5, num_results=5)
text_results_q5 = tindex.search(query_q5, num_results=5)

# Extract filenames into sets for easy comparison
vec_files_q5 = set(r["filename"] for r in vec_results_q5)
text_files_q5 = set(r["filename"] for r in text_results_q5)

# Find files that are in vector results but NOT in text results
diff_files = vec_files_q5 - text_files_q5
print(f"Files in vector results but not in text results: {diff_files}")

# ==========================================
# Q6. Hybrid search (RRF)
# ==========================================
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

query_q6 = "How do I give the model access to tools?"
v_q6 = embed.encode(query_q6)

# Get top 10 results from both methods to give RRF enough data to work with
vec_results_q6 = vindex.search(v_q6, num_results=10)
text_results_q6 = tindex.search(query_q6, num_results=10)

# Fuse the results
hybrid_results_q6 = rrf([vec_results_q6, text_results_q6])

print(f"First result filename after RRF: {hybrid_results_q6[0]['filename']}")
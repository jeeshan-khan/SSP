# flake8: noqa
import os
import google.generativeai as genai

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")
# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://vector-db:6333",
    collection_name="learning_vector",
    embedding=embeddings
)

async def process_query(query:str):
    print("Searching chunks for this particular query: ", query)
    
search_results = vector_db.similarity_search(
    query=query,
    k=3
)
context = "\n\n".join([
    f"""
Page Number: {result.metadata.get('page', 'Unknown')}

Page Content:
{result.page_content}
"""
    for result in search_results
])

# System Prompt
SYSTEM_PROMPT = f"""
You are a helpful AI assistant.

Answer the user's question ONLY from the provided PDF context.

If the answer is not present in the context, say:
"I could not find the answer in the PDF."

Always mention the page numbers where the answer was found.

Retrieved Context:
{context}
"""

# Generate Gemini Response
response = model.generate_content(
    SYSTEM_PROMPT + f"\n\nUser Question: {query}"
)

# Final Output
print("\n================ AI RESPONSE ================\n")
print(response.text)
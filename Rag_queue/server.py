from fastapi import FastAPI, Query
from .message_queue.worker import process_query
from .message_queue.connection import queue

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Server is running!"}

@app.post("/chat")
def chat(
    query: str = Query(..., description="user's query")
):
    job = queue.enqueue(process_query, query)

    return {
        "status": "Query received and being processed!",
        "job_id": job.id
    }
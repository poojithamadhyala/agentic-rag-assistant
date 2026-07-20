FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/
RUN python scripts/ingest.py --out .index
ENV RAG_INDEX_DIR=/app/.index
EXPOSE 8000
CMD ["uvicorn", "rag_agent.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

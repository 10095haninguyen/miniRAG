miniRAG - He thong RAG voi Google Gemini

Gioi thieu
He thong RAG (Retrieval-Augmented Generation) su dung:
- LLM: Google Gemini 2.5 Flash
- Vector Database: FAISS
- Embedding: all-MiniLM-L6-v2
- Retriever: Ensemble (Vector Search + BM25)

Cau hinh toi uu
- Chunk Size: 2200
- Chunk Overlap: 400
- So chunks lay (k): 3
- Trong so Ensemble: Vector 60% - BM25 40%

Cai dat
1. Clone repository
git clone https://github.com/10095haninguyen/miniRAG.git
cd miniRAG

2. Tao file api_key.env:
API_KEY=your_google_api_key_here

python test.py

10095haninguyen

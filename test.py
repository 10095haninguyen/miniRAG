import os
import time
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

load_dotenv(r"d:\miniRAG\api_key.env")
api_key = os.getenv("API_KEY")

os.environ["GOOGLE_API_KEY"] = api_key


TEXTS_CACHE = "D:/miniRAG/texts_cache.pkl"
FAISS_INDEX_PATH = "D:/miniRAG/faiss_index"

if os.path.exists(TEXTS_CACHE):
    with open(TEXTS_CACHE, 'rb') as f:
        texts = pickle.load(f)
else:
    loader = PyPDFLoader("D:/miniRAG/LSDCSVN.pdf")
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2200, chunk_overlap=400)
    texts = text_splitter.split_documents(document)
    with open(TEXTS_CACHE, 'wb') as f:
        pickle.dump(texts, f)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder="./embedding_cache")

os.makedirs("./embedding_cache", exist_ok=True)

if os.path.exists(FAISS_INDEX_PATH):
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
start = time.time()
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3

ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever, bm25_retriever],
    weights=[0.6, 0.4]  
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,  
    return_source_documents=True
)

query = "TÌNH HÌNH THẾ GIỚI VÀ VIỆT NAM CUỐI THẾ KỶ XIX ĐẦU THẾ KỶ XX"
response = qa_chain.invoke({"query": query})
print(f"{response['result']}")

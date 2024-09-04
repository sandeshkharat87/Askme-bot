from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")


DB = FAISS.load_local(folder_path="data/database/estatements-vectors",embeddings=embed_model,allow_dangerous_deserialization=True)

for i in DB.similarity_search("15000 spent"):
    print(i)
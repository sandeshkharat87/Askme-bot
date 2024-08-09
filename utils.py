from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables  import RunnablePassthrough
from langchain_community.vectorstores import FAISS
import shutil as sh
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import os
import sys

print("--Start--")

## PATHS
store_path = "store"
database_path = os.path.join(store_path,"DATABASE")
docs_path = os.path.join(store_path,"DOCS")
processed_docs_path = os.path.join(store_path,"processed")
os.makedirs(database_path,exist_ok=True)
os.makedirs(docs_path,exist_ok=True)
os.makedirs(processed_docs_path,exist_ok=True)




# create docs
def _create_docs():
    pdf_loader = DirectoryLoader(docs_path,glob = "**/*.pdf",loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(docs_path,glob = "**/*.txt")
    docx_loader = DirectoryLoader(docs_path,glob = "**/*.docx")
    
    all_loader = [pdf_loader,txt_loader]

    docs_collection = []
    for myloader in all_loader:
        docs_collection.extend(myloader.load())

    ### split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap=150, length_function=len)

    DOCS = splitter.split_documents(docs_collection)


    return DOCS

# add docs to database
def add_docs_to_database(DB,EMBED_MODEL):
    temp_docs = _create_docs()
    if len(temp_docs)>0:
        temp_db = FAISS.from_documents(temp_docs,EMBED_MODEL)

        ## if database is empty create new else merge to old
        if  os.path.exists(os.path.join(database_path,"index.pkl")) and os.path.exists(os.path.join(database_path,"index.faiss")):
            print("--merging to crt database")
            current_db = DB
            current_db.merge_from(temp_db)
            current_db.save_local(database_path)
            ## Move files rom DOCS -> Processed Docs
            remove_docs()
            print("Removed docs")
            print("--Saved new docs--")
        else:
            print("--Fresh--")
            temp_db.save_local(database_path)
            ## Move files rom DOCS -> Processed Docs
            remove_docs()
    else:
        print("Nothing to ADD, DOCS is empty", len(temp_docs))
        

# Move docs from DOCS - Processed
def remove_docs():
    for i in os.listdir(docs_path):
        sh.move(os.path.join(docs_path,i), 
                os.path.join(processed_docs_path,os.path.basename(i)))







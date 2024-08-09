from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables  import RunnablePassthrough
from langchain_community.vectorstores import FAISS
import shutil as sh
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import os
import sys
import streamlit as st
import os
import pandas as pd
#utils
from utils import add_docs_to_database, database_path,docs_path,processed_docs_path
EMBED_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Database
DB = None

### Constants
# LLMS
model_name = "mistralai/Mistral-7B-v0.1"
zypher_model = "HuggingFaceH4/zephyr-7b-beta"
LLM = HuggingFaceHub(repo_id=zypher_model,huggingfacehub_api_token = "XXXXXXXXXXXXXXXXXXXXXX",model_kwargs={"return_full_text" : False})

# load saved database
def load_saved_database():
    global DB
    if  os.path.exists(os.path.join(database_path,"index.pkl")) and os.path.exists(os.path.join(database_path,"index.faiss")):
        print("Inside load db index.pkl & index.faiss file EXISTS ")
        DB = FAISS.load_local(database_path,EMBED_MODEL)
    else:
        print("Error while loading database please check")
        DB = None


load_saved_database()

def refresh_folder():
    return os.listdir("store/processed")
    
##### SideBar
with st.sidebar:
    FILES = st.file_uploader("Choose a file", accept_multiple_files=True, type=["pdf","docx","txt"])
    # FILES = st.file_uploader("Choose a file", type=["pdf","docx","txt"])

    btn = st.button("SAVE")
    if btn:
        for f in FILES:
            f_data = f.read()
            with open(f"store/DOCS/{f.name}","wb") as file:
                file.write(f_data)
        ### if btn pressed save docs to database    
        # Save Files in DOCS
        # Move Files from DOCS - Processed
        add_docs_to_database(DB,EMBED_MODEL)

    p_files = refresh_folder() 
    st.success("Available Documents")
    st.write(pd.DataFrame.from_dict({"FileNames":p_files}))



# Ask Question
def AskQuestion(Question,DB,LLM):
    if DB:
        RTVR = DB.as_retriever()
        myprompt = """
        Context : {context}

        You are a knowledge Bot. You have to give answers from the given
        context. If answers not in the context just say I Dont know.
        Dont try to makeup an answer.
        In Return Just Give Answer nothing else.

        Question: {question}

        Result:
        """
        prompt = PromptTemplate.from_template(myprompt)

        chain = ( {"context":RTVR , "question":RunnablePassthrough()}  | prompt | LLM | StrOutputParser())

        result = chain.invoke(Question)
        return result
    else:
        return ["Database is --EMPTY--"]

###### Main Page
Question = st.text_input("Enter Question Here....")
answer_btn = st.button("Answer")
if answer_btn:
    st.write(AskQuestion(Question,DB,LLM))




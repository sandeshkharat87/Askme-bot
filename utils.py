from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS
from langchain.agents import initialize_agent, AgentType,create_react_agent,AgentExecutor
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
# from langchain.runnables import 


embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
qa_temp = """
You are a helpful bot which answers question using context provided.
Finding answers from the context. If You couldnot find answer just say i dont know
Dont try to makeup an answer.

Context:{context}
Question:{question}
Answer:
"""
qa_prompt = PromptTemplate.from_template(qa_temp)

# QA = RetrievalQA.from_chain_type(llm=llm,   prompt=qa_temp)




##load envs
from dotenv import load_dotenv
load_dotenv()

# EMBED_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(model_name="llama-3.1-70b-versatile")
DB = FAISS.load_local(folder_path="data/database/estatements-vectors",embeddings=embed_model,allow_dangerous_deserialization=True)
QA = RetrievalQA.from_chain_type(
    llm, retriever=DB.as_retriever(), chain_type_kwargs={"prompt": qa_prompt}
)

prompt = hub.pull("hwchase17/react")


@tool
def web_search_tool(state):
    """Use this tool for finding inforamtion on the web/internet for current scenarios"""
    findings = DDGS().text(state,max_results=3)
    findings = " ".join([i["body"] for i in findings])
    return findings

@tool
def wether_tool(state):
    """Useful for finding wether of the current day"""    
    return "28 Degree Celcious"


@tool
def local_search_tool(state):
    """Useful for finding informaiton about bank statements, money """
    answer = QA(state)
    
    return answer["result"]
        

my_tools = [web_search_tool,wether_tool,local_search_tool]




agent = initialize_agent(llm=llm, tools = my_tools,verbose=True,handle_parsing_error=True)
# react_agent = create_react_agent(llm=llm, tools=my_tools,prompt=prompt)
# AE = AgentExecutor.from_agent_and_tools(llm=llm ,agent=react_agent,tools=my_tools,
                                        # verbose=True,handle_parsing_error=True)

def bot(question):
    # result = AE.invoke(dict(input=question))
    result = agent.invoke(dict(input=question))
    return result["output"]
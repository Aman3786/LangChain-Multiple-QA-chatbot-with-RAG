from langchain_community.embeddings import OllamaEmbeddings   # When Ollama Model Used
from langchain_community.llms import Ollama   # Used When Ollama installed Locally
from langchain_community.vectorstores import FAISS  #Facebook AI DB to Store Embedded vector
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

# Huggingface Import 
# from langchain_huggingface import HuggingFaceEndpoint
from langchain import hub
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")


hf_embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5", 
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

# Wikipedia Tool to search for any topic
wiki_api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki_tool=WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

# WebBaseLoader Tool to load all text from HTML webpages into a document
Webloader=WebBaseLoader("https://ollama.com/blog/embedding-models")
docs=Webloader.load()
Web_text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectordb=FAISS.from_documents(Web_text_splitter,hf_embeddings)
WebRetriever=vectordb.as_retriever()
WebRetreiver_tool = create_retriever_tool(WebRetriever, name="Ollama Embedding Model", description="Search for information about Ollama Embedding Model. For any questions about Ollama Embedding Model, you must use this tool!")


# PyPDFLoader Tool to load Pdf files into a document
Pdfloader=PyPDFLoader("TL.pdf")
Pdfdocuments=Pdfloader.load()
Pdf_text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_documents=Pdf_text_splitter.split_documents(Pdfdocuments)
Pdfvectordb=FAISS.from_documents(final_documents[:120],hf_embeddings)
PdfRetriever = Pdfvectordb.as_retriever()
PdfRetriever_tool = create_retriever_tool(PdfRetriever, name="Transfer Learning", description="Search for information about Transfer Learning. For any questions about Transfer Learning, you must use this tool!")

tools = [wiki_tool,WebRetreiver_tool,PdfRetriever_tool]


# OpenSource Llama3 Model available on Huuggingface
hf_llm=HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B",
    model_kwargs={"temperature":0.1,"max_length":300}
)

# llm = Ollama(model="llama2:7b")     #Local_llama2 model


# Open-Source Openai chat Prompt Template from langchain hub. Link : https://smith.langchain.com/hub/hwchase17/openai-functions-agent
prompt = hub.pull("hwchase17/openai-functions-agent")


agent=create_tool_calling_agent(hf_llm,tools,prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
user_input = input("Enter Search Query: ")
agent_executor.invoke({"input":user_input})
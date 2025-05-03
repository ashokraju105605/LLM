# Warning control
import warnings
warnings.filterwarnings('ignore')
## 
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.md import partition_md
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import dict_to_elements
## 
import chromadb
##

# Pre-process the pdf file
from langchain_community.document_loaders import UnstructuredFileLoader
loader = UnstructuredFileLoader(
    "bigita.pdf", strategy="fast", mode="elements"
)
docs = loader.load()
docs[:5] #sample check
filename = "your_pdf.pdf"
pdf_elements = partition_pdf(filename=filename)
## 

# Load the Documents into the Vector DB
elements = chunk_by_title(pdf_elements)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
documents = []
for element in elements:
    metadata = element.metadata.to_dict()
    del metadata["languages"]
    metadata["source"] = metadata["filename"]
    documents.append(Document(page_content=element.text, metadata=metadata))
## 
embeddings = OpenAIEmbeddings(api_key="your_key")
vectorstore = Chroma.from_documents(documents, embeddings)

## Set-up the retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

## prompt template
template = """You are an AI assistant for answering questions about the Bhagavad Gita document.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the document, politely inform them that you are tuned to only answer questions about the Bhagavad Gita.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
prompt = PromptTemplate(template=template, input_variables=["question", "context"])
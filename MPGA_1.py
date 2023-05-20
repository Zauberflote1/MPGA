# from dotenv import load_dotenv
import os
import os.path
import pinecone
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv()

loader = UnstructuredPDFLoader("h:/programs/vectordbproj/booka.pdf")
data = loader.load()
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')
print (texts)


openai_api_key=(os.getenv("OPENAI_API_KEY"))
pinecone_api_key=(os.getenv("PINECONE_API_KEY"))
pinecone_env=(os.getenv("PINECONE_ENV"))
embeddings = OpenAIEmbeddings(openai_api_key)

# initialize pinecone
pinecone.init(
    api_key = pinecone_api_key,  # find at app.pinecone.io
    environment = pinecone_env # next to api key in console
)

index_name = "mpga"
namespace = "bookai"

docsearch = Pinecone.from_texts(
  [t.page_content for t in texts], embeddings,
  index_name=index_name, namespace=namespace)

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
chain = load_qa_chain(llm, chain_type="stuff")
# 
query = "explain book A from aristotle"
docs = docsearch.similarity_search(query, namespace=namespace)

print(chain.run(input_documents=docs, question=query))

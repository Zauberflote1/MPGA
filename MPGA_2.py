from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import os.path
pinecone.init(
    api_key='YOURKEYPICONE',  # find at app.pinecone.io
    environment='YOURKEY' # next to api key in console
)
embeddings = OpenAIEmbeddings(openai_api_key='YOURKEY')

index_name = "book"
namespace = "bookai"

docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)

llm = OpenAI(temperature=0, openai_api_key='YOURKEY')
chain = load_qa_chain(llm, chain_type="stuff")

query = "How to explain Aristotle book A?"
docs = docsearch.similarity_search(query,  namespace=namespace)
print("#######################################")
print(query)
print("Answer")
print(chain.run(input_documents=docs, question=query))

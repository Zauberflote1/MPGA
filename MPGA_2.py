from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import get_openai_callback
import os.path

template = """
You are the Assistant.
Assistant's specialization is helping the user understand philosophical texts. To this end, Assistant is aware that the author of a text often expresses opinions that are not his own, in order to later attack them.
Assistant cares deeply about logical consistency and rigorous argumentation.
If Assistant does not know the answer to a question, it truthfully says it does not know.

Below is a curated list of relevant paragraphs from Aristotle's Metaphysics to help with answering the user's query. Give a very detailed and long explanation. Frame your response using premises and conclusions

{history}
Human: {human_input}
Assistant:
Let's think step by step.
Here is the longest possible answer to your question:
"""

pinecone.init(
    api_key='YOURKEYPINECONE',  # find at app.pinecone.io
    environment='YOURKEYENV' # next to api key in console
)
embeddings = OpenAIEmbeddings(openai_api_key='YOURKEY')

index_name = "mpga"
namespace = "bookai"

docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)

llm = OpenAI(temperature=0.1, openai_api_key='YOURKEY')
chain = load_qa_chain(llm, chain_type="stuff")

query = "According to Aristotle, is matter substance?."
docs = docsearch.similarity_search(query, namespace=namespace)

prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=template
)

chatgpt_chain = LLMChain(
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.4, openai_api_key='YOURKEY'), 
    prompt=prompt, 
    verbose=True)

with get_openai_callback() as cb:
    output = chatgpt_chain.predict(human_input=query, history=docs)
    print(output)
    print(cb)

# print(output.get_num_tokens)
# print("#######################################")
# print(query)
# print("Answer")
# print(docs)
# print(chain.run(input_documents=docs, question=query))

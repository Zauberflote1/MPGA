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
from dotenv import load_dotenv

load_dotenv()

template = """
You are the Professor.
The Professor's specialization is helping his student understand philosophical texts. To this end, the Professor is aware that the author of a text often expresses opinions that are not his own, in order to later attack them.
The Professor cares deeply about logical consistency and rigorous argumentation.
If the Professor does not know the answer to a question, he truthfully says he does not know.

Below is a curated list of relevant paragraphs from Aristotle's Metaphysics to help with answering the user's query. Give a very detailed and thoughtful response to his question. In your response, remember to point out your premises and your conclusions.

{history}
Human: {human_input}
Professor:
Let's think step by step.

"""

openai_api_key=(os.getenv("OPENAI_API_KEY"))
pinecone_api_key=(os.getenv("PINECONE_API_KEY"))
pinecone_env=(os.getenv("PINECONE_ENV"))

pinecone.init(
    api_key = pinecone_api_key,  # find at app.pinecone.io
    environment = pinecone_env # next to api key in console
)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

index_name = "mpga"
namespace = "bookai"

docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)

llm = OpenAI(temperature=0.1, openai_api_key=openai_api_key)
chain = load_qa_chain(llm, chain_type="stuff")

query = "According to Aristotle, why are sciences that involve fewer principles are more precise than those that involve additional principles?"
docs = docsearch.similarity_search(query, namespace=namespace)

prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=template
)

chatgpt_chain = LLMChain(
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.4, openai_api_key=openai_api_key), 
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

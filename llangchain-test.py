from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=model_path,
    max_tokens=256,
    n_gpu_layers=25,
    n_batch=256,
    callback_manager=callback_manager,
    n_ctx=1024,
    verbose=True,
)

from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os

loader = PyPDFLoader("./data/files/H-AES - Towards Automated Essay Scoring for Hindi.pdf")

data = loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

docs=text_splitter.split_documents(data)

len(docs)

docs[0]


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '105b615f-b9b1-43e6-a92f-324d3af15bbe')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "mined" # put in the name of your pinecone index here

docsearch = Pinecone.from_existing_index(index_name, embeddings)

query="What corpora are used for H-AES?"

docs=docsearch.similarity_search(query)

chain=load_qa_chain(llm, chain_type="stuff")
chain.run(input_documents=docs, question=query)
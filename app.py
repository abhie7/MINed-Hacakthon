import os
import shutil
import tempfile
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from main_parser import MainParser
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer
import pinecone
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfFileReader
from docx import Document as DocxDocument

def parse_pdf(file_path):
    with open(file_path, 'rb') as f:
        pdf_reader = PdfFileReader(f)
        text = ''
        for page_num in range(pdf_reader.getNumPages()):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

def parse_docx(file_path):
    docx_document = DocxDocument(file_path)
    text = ' '.join([paragraph.text for paragraph in docx_document.paragraphs])
    return text

def fetch_document_content(doc_ids):
    # Implement a function to fetch content of documents using their IDs
    # This function should return a list of document contents
    pass

app = Flask(__name__, template_folder="./templates")

model_path = "/Users/abhiraj/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGML/snapshots/3140827b4dfcb6b562cd87ee3d7f07109b014dd0/llama-2-13b-chat.ggmlv3.q5_1.bin"

# Initialize the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize LangChain LLM
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

# Initialize Pinecone
pinecone_api_key = os.environ.get('PINECONE_API_KEY', '105b615f-b9b1-43e6-a92f-324d3af15bbe')
pinecone_api_env = os.environ.get('PINECONE_API_ENV', 'gcp-starter')
index_name = "mined"
pinecone.init(api_key=pinecone_api_key, environment=pinecone_api_env)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
docsearch = pinecone.Index(index_name, embeddings)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        query = request.json['userInput']
        query_vector = model.encode([query])[0].tolist()
        query_results = docsearch.query(query_vector, top_k=5)
        print(f"query_results: {query_results}")  # Add this line
        # Extract the IDs from the matches
        doc_ids = [match['id'] for match in query_results['matches']]
        print(f"doc_ids: {doc_ids}")  # Add this line
        # Fetch document content
        docs = fetch_document_content(doc_ids)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        return jsonify({'result': response})
    except Exception as e:
        print(f"Exception occurred: {e}")  # Log the exception
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(file_path)

        # Process the files in the directory
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            _, file_extension = os.path.splitext(file_path)
            if file_extension == '.pdf':
                text = parse_pdf(file_path)
            elif file_extension == '.docx':
                text = parse_docx(file_path)
            else:
                continue

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            docs = text_splitter.split_documents(text)

            # Use Llama model to generate response
            userInput = request.json['userInput']
            userInput_vector = model.encode([userInput])[0].tolist()
            query_results = docsearch.query(userInput_vector, top_k=5)
            print(f"query_results: {query_results}")  # Add this line
            # Extract the IDs from the matches
            doc_ids = [match['id'] for match in query_results['matches']]
            print(f"doc_ids: {doc_ids}")  # Add this line
            # Fetch document content
            docs = fetch_document_content(doc_ids)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=userInput)

            print(f"Response from chain.run(): {response}")  # Add this line

            return jsonify({'result': response}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app.run(debug=True)

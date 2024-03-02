# MINeD Hackathon 2024 - Project Documentation

## Project Overview

### Project Definition
**Project Name:** ChatWithAnyScientificDocument

### Project Goals Achieved:
1. **Implemented a Versatile Document Parser:** Developed a Document Parser capable of handling multiple document types including `PDF, TXT, HTML, DOCX, .TEX, and PPT`.

2. **Developed a Chat Interface:** Created a user-friendly chat interface using `HTML, CSS, JS, and Flask`. This interface allows users to upload their document and search using their required query.

3. **User Input Question Answering Functionality:** Developed a functionality within the chat interface to enable users to input questions related to the parsed document content.

## Installation and Setup

### Installing Dependencies

To install the required dependencies, you can use the `requirements.txt` file provided. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

### Downloading and Setting up the Model

```python
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
lcpp_llm = Llama(
  model_path=model_path,
  n_threads=2,
  n_batch=512,
  n_gpu_layers=32
)
```

# Usage
### Text Generation
```python
prompt = "Summarize the 'Attention is all you need' research paper in 100 words."
prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.
USER: {prompt}
ASSISTANT:
'''

response=lcpp_llm(prompt=prompt_template, max_tokens=256, temperature=0.5, top_p=0.95,
         repeat_penalty=1.2, top_k=150,
         echo=True)

print(response)
print(response["choices"][0]["text"][len(prompt_template):])
```

### Question Answering
```python
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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

query="What corpora are used for H-AES?"
docs=docsearch.similarity_search(query)
chain=load_qa_chain(llm, chain_type="stuff")
chain.run(input_documents=docs, question=query)

```

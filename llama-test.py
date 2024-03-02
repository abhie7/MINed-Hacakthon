from huggingface_hub import hf_hub_download
from llama_cpp import Llama

model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGUF"  # Change this to the GGUF model name
model_basename = "llama-2-7b-chat.Q4_0.gguf"  # Change this to the GGUF model basename

# model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
model_path = "/Users/abhiraj/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/llama-2-7b-chat.Q4_0.gguf"
print(model_path)

lcpp_llm = None
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=4,  # CPU cores
    n_batch=256,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=2  # Change this value based on your model and your GPU VRAM pool.
)

prompt = "Summarize the 'Attention is all you need' research paper in 250 words."
prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.

USER: {prompt}

ASSISTANT:
'''

response=lcpp_llm(prompt=prompt_template, max_tokens=256, temperature=0.5, top_p=0.95,
    repeat_penalty=1.2, top_k=150,
    echo=True)

print("Answer:\n", response["choices"][0]["text"][len(prompt_template):])

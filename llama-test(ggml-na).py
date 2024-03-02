from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"  # Change this to the GGUF model name
# model_basename = "llama-2-13b-chat.Q5_K_M.gguf"  # Change this to the GGUF model basename

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q4_K_M.bin"

# model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
# model_path = "/Users/abhiraj/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGUF/snapshots/4458acc949de0a9914c3eab623904d4fe999050a/llama-2-13b-chat.Q5_K_M.gguf"
# model_path = "/Users/abhiraj/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGML/snapshots/3140827b4dfcb6b562cd87ee3d7f07109b014dd0/llama-2-13b-chat.ggmlv3.q5_1.bin"
model_path = "/Users/abhiraj/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGML/snapshots/3140827b4dfcb6b562cd87ee3d7f07109b014dd0/llama-2-13b-chat.ggmlv3.q4_K_M.bin"
print(model_path)

lcpp_llm = None
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,  # CPU cores
    n_batch=64,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=1  # Change this value based on your model and your GPU VRAM pool.
)

prompt = "Summarize the 'Attention is all you need' research paper in 250 words."
prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.

USER: {prompt}

ASSISTANT:
'''

response=lcpp_llm(prompt=prompt_template, max_tokens=128, temperature=0.5, top_p=0.95,
    repeat_penalty=1.2, top_k=150,
    echo=True)

print("Answer:\n", response["choices"][0]["text"][len(prompt_template):])

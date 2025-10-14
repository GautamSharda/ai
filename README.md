## todo
increase the space in the network volume, maybe make a .toml, idk

get the qwen thinking fp 8 working, run eval, do submission, then maybe consider dpseek 3.1 terminus and gpt oss


old: 
benchmark qwen3-0.6b on (1) arc-agi-2 eval set && (2) perf vllm vs hf transformers && tinygrad (maybe also vs torch)

qwen-next
dsv3.2
flash attention
determinism
lora/qlora


|| || |Transformer|N⋅H⋅2⋅L⋅D⋅S|-| |GQA/MQA|N⋅G⋅2⋅L⋅D⋅S|H→G|

N : Model Layer

H : Attention Head per Layer

G : Key/Value Head Number in GQA or MQA

L : Sequece Length

D : Dimesion of each head

S : K/V bytes (no quantization is 2, 1 for fp8, 0.5 for q_4)

So for Qwen3-32B

64*8*2*1024*128*2 = 268435456 = 0.25G

1K context need 0.25G

0.25 GB / 1k context 

At FP8 = 1 byte per weight --> 80 billion param = 80 GB

Does this work? 16k tokens, 6-8 tok/s, let's say 7, we want to max out tokens, input tokens for a problem is  

● (1) Server start command:

export HF_HOME=/ai_network_volume/huggingface_cache && export HF_HUB_CACHE=/ai_network_volume/huggingface_cache/hub && vllm serve Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 --max-model-len 32768 --enforce-eager --reasoning-parser deepseek_r1 --tensor-parallel-size 4 --disable-custom-all-reduce --gpu-memory-utilization 0.95

try 32768 / 2

  (2) Inference command:

  curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8", "messages": [{"role": "user", "content": "Hi!"}]}'


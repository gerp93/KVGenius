# Complete Guide: How AI Language Models Work
## Understanding All the Pieces and How They Fit Together

---

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Core Components](#core-components)
3. [The Model Lifecycle](#the-model-lifecycle)
4. [Hardware Stack](#hardware-stack)
5. [Software Stack](#software-stack)
6. [How Inference Works](#how-inference-works)
7. [Memory and Performance](#memory-and-performance)
8. [Your Project Architecture](#your-project-architecture)

---

## The Big Picture

### What is a Language Model?
A language model is essentially a **giant mathematical function** that:
1. Takes text as input (your question)
2. Processes it through billions of learned parameters (weights)
3. Predicts the most likely next words to generate a response

Think of it like an incredibly sophisticated autocomplete system that understands context, grammar, facts, and reasoning.

### The Journey of a Chat Message

```
Your Text Input
    ↓
Tokenization (text → numbers)
    ↓
Model Weights (billions of parameters)
    ↓
GPU Computation (trillions of math operations)
    ↓
Probability Distribution (which word is most likely next?)
    ↓
Decoding (numbers → text)
    ↓
Response Text
```

---

## Core Components

### 1. The Model Weights (The "Brain")

**What they are:**
- Billions of numbers (parameters) stored in files
- Each number represents a learned connection in the neural network
- For DeepSeek-Coder-6.7B: 6.7 billion parameters ≈ 13-27 GB on disk

**File formats:**
- `.safetensors` - Safer, faster format (we use this)
- `.bin` - Older PyTorch format
- `.ckpt` - Checkpoint format

**Where they live:**
```
model_cache/
└── models--deepseek-ai--deepseek-coder-6.7b-instruct/
    └── snapshots/
        └── <commit-hash>/
            ├── model-00001-of-00002.safetensors  ← Part 1 of weights
            ├── model-00002-of-00002.safetensors  ← Part 2 of weights
            ├── config.json                        ← Model architecture
            └── tokenizer.json                     ← Vocabulary mapping
```

**What happens when you load a model:**
1. Read config.json → understand model architecture
2. Load .safetensors files → load all 6.7 billion parameters into RAM/VRAM
3. Move tensors to GPU → now ready for computation

---

### 2. The Tokenizer (Text ↔ Numbers Translator)

**What it does:**
Converts between human text and numbers the model understands.

**Example tokenization:**
```
Input:  "Hello, how are you?"
Tokens: [15496, 11, 703, 366, 291, 30]
        ↑      ↑   ↑   ↑   ↑   ↑
      Hello   ,  how are you  ?
```

**Why tokenization matters:**
- Models only work with numbers, not text
- Each token is a word, part of a word, or punctuation
- Vocabulary size: typically 30,000-100,000 tokens
- DeepSeek-Coder vocabulary: 32,000 tokens

**Token budget:**
- Models have a maximum context length (e.g., 4096 tokens)
- That's ~3000 words of conversation history
- Older messages get truncated to stay under limit

---

### 3. The Model Architecture (The Network Structure)

**Transformer Architecture (simplified):**
```
Input Tokens
    ↓
Embedding Layer (convert token IDs to dense vectors)
    ↓
┌─────────────────────────────────┐
│  Transformer Block 1            │
│  - Self-Attention (what to focus on?)
│  - Feed-Forward (process information)
├─────────────────────────────────┤
│  Transformer Block 2            │
│  ... (repeat 32 times for 6.7B model)
├─────────────────────────────────┤
│  Transformer Block 32           │
└─────────────────────────────────┘
    ↓
Output Layer (probability for each possible next token)
    ↓
Sampling/Decoding (pick the next token)
```

**Key concepts:**
- **Layers:** DeepSeek-Coder-6.7B has 32 transformer layers
- **Hidden size:** 4096 dimensions per layer
- **Attention heads:** 32 heads per layer (parallel attention mechanisms)
- **Parameters:** Most are in the feed-forward networks in each layer

---

### 4. The Configuration (`config.json`)

**What it contains:**
```json
{
  "model_type": "llama",              ← Architecture type
  "hidden_size": 4096,                ← Vector dimensions
  "num_hidden_layers": 32,            ← Number of transformer blocks
  "num_attention_heads": 32,          ← Attention mechanism heads
  "vocab_size": 32000,                ← Tokenizer vocabulary size
  "max_position_embeddings": 4096,    ← Maximum context length
  ...
}
```

**Why it's important:**
- Tells PyTorch exactly how to construct the model
- Without it, we wouldn't know how to interpret the weight files
- Must match the weights exactly or loading fails

---

## The Model Lifecycle

### Phase 1: Pre-training (Done by DeepSeek)
```
Massive Text Dataset (trillions of words)
    ↓
Supercomputer Cluster (thousands of GPUs)
    ↓
Months of Training
    ↓
Base Model Weights
```

**What happens:**
- Model learns language patterns, facts, reasoning
- Cost: millions of dollars in compute
- Result: 6.7 billion parameters that capture human knowledge

### Phase 2: Fine-tuning (Done by DeepSeek)
```
Base Model
    ↓
Instruction Dataset (Q&A pairs, code examples)
    ↓
Additional Training
    ↓
Instruct Model (deepseek-coder-6.7b-instruct)
```

**What happens:**
- Model learns to follow instructions
- Learns conversational format
- Specialized for coding tasks (DeepSeek-Coder)

### Phase 3: Inference (What You Do)
```
Download Pre-trained Weights
    ↓
Load into GPU Memory
    ↓
Send User Input
    ↓
Generate Response
    ↓
Repeat
```

**What happens:**
- No training, just using the model
- Much cheaper (single GPU, seconds not months)
- This is what our chatbot does

---

## Hardware Stack

### CPU vs GPU: Why GPU Matters

**CPU (Central Processing Unit):**
- Designed for general-purpose computing
- Executes instructions sequentially
- Good at: complex logic, branching, control flow
- 8-16 cores typically

**GPU (Graphics Processing Unit):**
- Designed for parallel computation
- Thousands of small cores working simultaneously
- Good at: matrix multiplication (exactly what neural networks do!)
- Your RTX 5070 Ti: 8,960 CUDA cores

**Why LLMs need GPUs:**
```
Matrix Multiplication Example:
- 4096 x 4096 matrix multiply = 68 billion operations
- CPU: ~1 second (sequential)
- GPU: ~0.01 seconds (parallel)
- Speedup: 100x faster!
```

**For a single model inference:**
- Thousands of matrix multiplications
- CPU: minutes per response
- GPU: 2-5 seconds per response

---

### GPU Architecture (Your RTX 5070 Ti)

```
┌─────────────────────────────────────────┐
│  GPU Die                                │
│  ┌─────────────────────────────────┐   │
│  │  Streaming Multiprocessors (SMs)│   │
│  │  - 8,960 CUDA Cores             │   │
│  │  - Tensor Cores (AI acceleration)│   │
│  │  - RT Cores (ray tracing)        │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  VRAM (Video Memory)            │   │
│  │  - 16 GB GDDR7                  │   │
│  │  - High bandwidth (672 GB/s)    │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
         ↑
    PCIe 5.0 x16 (to CPU/System RAM)
```

**Key specs:**
- **CUDA Cores:** 8,960 (parallel processors)
- **VRAM:** 16 GB (where model weights live during inference)
- **Memory Bandwidth:** 672 GB/s (how fast data moves)
- **Compute Capability:** sm_120 (Blackwell architecture)

---

### Memory Hierarchy

```
Speed ↑     Size ↓     Cost ↑

GPU Registers      (fastest, tiny, per-core)
    ↓
GPU L1 Cache       (very fast, small, per-SM)
    ↓
GPU L2 Cache       (fast, medium, shared)
    ↓
VRAM (GPU Memory)  (fast, 16 GB, where model lives)
    ↓
System RAM         (slower, 32+ GB, temporary storage)
    ↓
NVMe SSD           (medium, 1+ TB, model storage)
    ↓
HDD                (slowest, TBs, archive)
```

**Where model weights live:**
1. **On disk (SSD):** `model_cache/` folder (~13 GB)
2. **During loading:** Briefly in System RAM
3. **During inference:** VRAM on GPU (16 GB)
4. **During computation:** Streaming through L2/L1 cache

---

## Software Stack

### The Full Stack (Bottom to Top)

```
┌─────────────────────────────────────────┐
│  Your Application (cli_app.py)         │  ← What you interact with
├─────────────────────────────────────────┤
│  Our Code (src/chat/, src/models/)     │  ← Our chatbot logic
├─────────────────────────────────────────┤
│  Transformers Library (Hugging Face)   │  ← Model loading, inference
├─────────────────────────────────────────┤
│  PyTorch (torch)                        │  ← Tensor operations, neural nets
├─────────────────────────────────────────┤
│  CUDA Runtime (cudart)                  │  ← GPU programming interface
├─────────────────────────────────────────┤
│  NVIDIA Driver                          │  ← Talks to GPU hardware
├─────────────────────────────────────────┤
│  GPU Hardware (RTX 5070 Ti)            │  ← Physical chip
└─────────────────────────────────────────┘
```

---

### PyTorch: The Foundation

**What PyTorch is:**
- Deep learning framework (like TensorFlow, JAX)
- Provides tensor operations (n-dimensional arrays)
- Automatic differentiation (for training, not needed for inference)
- GPU acceleration via CUDA

**Tensors:**
```python
# A tensor is just a multi-dimensional array
import torch

# 1D tensor (vector)
x = torch.tensor([1, 2, 3])  # shape: (3,)

# 2D tensor (matrix)
y = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)

# 3D tensor (batch of matrices)
z = torch.randn(32, 4096, 4096)  # shape: (32, 4096, 4096)
                                  # 32 batches of 4096x4096 matrices
```

**Why PyTorch for LLMs:**
- Efficient tensor operations on GPU
- Automatic memory management
- Pretrained model ecosystem
- Dynamic computation graphs (easier to debug)

---

### CUDA: GPU Programming

**What CUDA is:**
- NVIDIA's parallel computing platform
- Allows programs to use GPU for computation
- PyTorch uses CUDA under the hood

**CUDA Kernels:**
- Small programs that run on GPU
- Each kernel runs on thousands of threads simultaneously
- Example: matrix multiplication kernel

**Compute Capability (sm_120):**
- Defines what GPU features are available
- Each GPU generation has a compute capability version
- Kernels must be compiled for specific compute capabilities
- **This is why we have the sm_120 problem!**

**The sm_120 Issue Explained:**
```
PyTorch 2.5.1 was compiled with kernels for:
sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90

Your RTX 5070 Ti has compute capability:
sm_120 (not in the list!)

Result: "no kernel image is available"
```

---

### Hugging Face Transformers

**What it is:**
- Library for using pretrained models
- Handles model downloading, loading, inference
- Standardized interface for hundreds of models

**Key classes:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Automatically detects model type from config
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModelForCausalLM.from_pretrained("model-name")

# Inference
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0])
```

**What it does for you:**
- Downloads models from Hugging Face Hub
- Caches models locally
- Handles tokenization
- Provides generation methods (sampling, beam search, etc.)
- Manages attention masks, padding, etc.

---

## How Inference Works

### Step-by-Step Inference Flow

#### 1. Tokenization
```python
user_input = "What is Python?"
input_ids = tokenizer.encode(user_input)
# [1841, 387, 13218, 30]  (example token IDs)
```

#### 2. Create Input Tensor
```python
input_tensor = torch.tensor([input_ids]).to("cuda")
# Shape: (1, 4) - batch size 1, sequence length 4
```

#### 3. Forward Pass Through Model
```python
with torch.no_grad():  # No gradients needed (not training)
    outputs = model(input_tensor)
    # outputs.logits shape: (1, 4, 32000)
    # For each of 4 positions, probability over 32k tokens
```

**What happens inside:**
```
For each of 32 transformer layers:
  1. Self-Attention:
     - Q = input @ Wq  (Query projection)
     - K = input @ Wk  (Key projection)
     - V = input @ Wv  (Value projection)
     - Attention = softmax(Q @ K^T / sqrt(d)) @ V
     Result: Model "attends" to relevant parts of input
  
  2. Feed-Forward:
     - hidden = ReLU(input @ W1)
     - output = hidden @ W2
     Result: Process attended information
  
  3. Add & Normalize:
     - output = LayerNorm(input + output)
     Result: Stability and gradient flow
```

#### 4. Sample Next Token
```python
# Get logits for last position (what comes next?)
next_token_logits = outputs.logits[0, -1, :]  # Shape: (32000,)

# Apply temperature (controls randomness)
next_token_logits = next_token_logits / temperature

# Convert to probabilities
probs = torch.softmax(next_token_logits, dim=-1)

# Sample from distribution (or take argmax for greedy)
next_token = torch.multinomial(probs, num_samples=1)
# next_token = 15234 (example)
```

#### 5. Decode Token
```python
next_word = tokenizer.decode([next_token])
# "Python"
```

#### 6. Repeat Until Done
```python
generated_tokens = [next_token]
while next_token != EOS_TOKEN and len(generated_tokens) < max_length:
    # Append to input
    input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
    
    # Forward pass again
    outputs = model(input_tensor)
    
    # Sample next token
    next_token = sample(outputs.logits[0, -1, :])
    generated_tokens.append(next_token)

# Decode all tokens
response = tokenizer.decode(generated_tokens)
```

---

### Generation Strategies

#### Greedy Decoding (Deterministic)
```python
# Always pick the most likely token
next_token = torch.argmax(probs)
```
- **Pros:** Fast, reproducible
- **Cons:** Repetitive, boring responses

#### Sampling with Temperature
```python
# Temperature controls randomness
logits = logits / temperature  # temperature = 0.7

# temperature < 1.0: more confident (less random)
# temperature = 1.0: use raw probabilities
# temperature > 1.0: more random
```

#### Top-k Sampling
```python
# Only consider top k most likely tokens
top_k_probs, top_k_indices = torch.topk(probs, k=50)
# Sample from top 50 tokens only
```

#### Top-p (Nucleus) Sampling
```python
# Sample from smallest set of tokens whose cumulative prob > p
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumsum = torch.cumsum(sorted_probs, dim=-1)
mask = cumsum > p  # p = 0.9 (90% probability mass)
# Sample from this subset
```

**Our chatbot uses:**
```yaml
temperature: 0.7      # Slight randomness
top_k: 50             # Consider top 50 tokens
top_p: 0.9            # Nucleus sampling
do_sample: true       # Enable sampling (not greedy)
```

---

## Memory and Performance

### Memory Requirements

#### Model Weights
```
6.7 billion parameters × 2 bytes (float16) = 13.4 GB
6.7 billion parameters × 4 bytes (float32) = 26.8 GB
```

**We use float16 (half precision) to save memory.**

#### Activation Memory (During Inference)
```
Batch size × Sequence length × Hidden size × Num layers × 4 bytes
1 × 1024 × 4096 × 32 × 4 = 512 MB (approximate)
```

**This is temporary memory needed during computation.**

#### KV Cache (Key-Value Cache)
```
2 × Batch × Heads × Seq length × Head dim × Num layers × 2 bytes
2 × 1 × 32 × 1024 × 128 × 32 × 2 = 512 MB (approximate)
```

**Stores attention keys/values to avoid recomputing them.**

#### Total VRAM Usage
```
Model weights:   ~13.4 GB
Activations:     ~0.5 GB
KV Cache:        ~0.5 GB (grows with context)
PyTorch overhead: ~1.0 GB
─────────────────────────
Total:           ~15.4 GB (fits in 16 GB!)
```

---

### Performance Metrics

#### Tokens per Second (Throughput)
```
RTX 5070 Ti (after sm_120 fix): ~20-40 tokens/sec
RTX 4090: ~40-60 tokens/sec
H100 (datacenter): ~100-200 tokens/sec
CPU (slow): ~2-5 tokens/sec
```

**What affects speed:**
- GPU memory bandwidth (higher = faster)
- Compute power (more CUDA cores = faster)
- Model size (larger = slower)
- Sequence length (longer = slower)
- Batch size (bigger = more efficient per token)

#### Latency (Time to First Token)
```
Model loading: ~2-5 seconds (one time)
First token:   ~0.5-1 second
Subsequent:    ~0.025-0.05 seconds each (20-40 tokens/sec)
```

**For a 100-word response:**
- 100 words ≈ 130 tokens
- Time: ~3-6 seconds total

---

## Your Project Architecture

### File Structure Overview

```
KVGenius/
├── config/
│   └── config.yaml              ← Model settings, generation params
├── src/
│   ├── models/
│   │   └── model_loader.py      ← Loads model and tokenizer
│   ├── chat/
│   │   └── chatbot.py           ← Manages conversation, generates responses
│   └── utils/
│       └── config_helper.py     ← Reads config, saves history
├── model_cache/                 ← Downloaded model weights
│   └── models--deepseek-ai--deepseek-coder-6.7b-instruct/
├── cli_app.py                   ← Command-line interface
├── web_app.py                   ← Web interface (Flask)
├── inspect_model.py             ← Check model metadata
└── requirements.txt             ← Python dependencies
```

---

### Code Flow (Simplified)

#### 1. Application Startup (`cli_app.py`)
```python
# Load configuration
config = load_config()

# Initialize model loader
loader = ModelLoader(
    model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
    cache_dir="./model_cache",
    device="cuda",
    token=HUGGINGFACE_TOKEN
)

# Load model and tokenizer
tokenizer, model = loader.load_model()

# Create chatbot
chatbot = ChatBot(model, tokenizer, config)
```

#### 2. User Input
```python
while True:
    user_input = input("You: ")
    if user_input == "quit":
        break
```

#### 3. Generate Response (`chatbot.py`)
```python
def generate_response(self, user_input):
    # Add to conversation history
    self.conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Build prompt from history
    prompt = self.build_prompt()
    
    # Tokenize
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    # Generate
    outputs = self.model.generate(
        **inputs,
        max_length=self.max_length,
        temperature=self.temperature,
        top_k=self.top_k,
        top_p=self.top_p,
        do_sample=True
    )
    
    # Decode
    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract bot response (remove prompt)
    bot_response = response[len(prompt):]
    
    # Add to history
    self.conversation_history.append({
        "role": "assistant",
        "content": bot_response
    })
    
    return bot_response
```

#### 4. Display Response
```python
print(f"Bot: {response}")
```

---

### Configuration Options Explained

```yaml
model:
  name: "deepseek-ai/deepseek-coder-6.7b-instruct"
  # ↑ Which model to use from Hugging Face
  
  cache_dir: "./model_cache"
  # ↑ Where to store downloaded models (so you don't re-download)
  
  device: "cuda"
  # ↑ "cuda" = GPU, "cpu" = CPU, "auto" = choose automatically
  
  token: null
  # ↑ Hugging Face auth token (for private/gated models)

generation:
  max_length: 1000
  # ↑ Maximum total tokens (input + output)
  
  temperature: 0.7
  # ↑ Randomness (0.1 = very focused, 1.0 = balanced, 2.0 = very random)
  
  top_k: 50
  # ↑ Only consider top 50 most likely tokens
  
  top_p: 0.9
  # ↑ Nucleus sampling (consider tokens with 90% cumulative probability)
  
  repetition_penalty: 1.2
  # ↑ Discourage repeating words (1.0 = no penalty, higher = less repetition)
  
  do_sample: true
  # ↑ true = sample from distribution, false = always pick most likely (greedy)

chat:
  max_history: 5
  # ↑ How many conversation turns to remember (avoids context length issues)
  
  system_prompt: "You are a helpful AI assistant."
  # ↑ Instructions given to model at start of conversation

app:
  save_history: true
  # ↑ Save conversations to JSON file
  
  history_file: "./chat_history.json"
  # ↑ Where to save conversation history
```

---

### Data Flow Diagram

```
User Input: "Write a Python function"
    ↓
Conversation History: [
  {role: "system", content: "You are a helpful AI assistant."},
  {role: "user", content: "Write a Python function"}
]
    ↓
Build Prompt: "<system>You are...</system><user>Write...</user><assistant>"
    ↓
Tokenizer: [1, 887, 366, 263, 8444, 319, 20255, ...]
    ↓
Model (6.7B params on GPU):
  - 32 transformer layers
  - Attention mechanisms
  - Feed-forward networks
    ↓
Output Logits: [0.001, 0.003, 0.021, ..., 0.156, ...]
    ↓
Sampling (temperature, top_k, top_p):
  - Apply temperature scaling
  - Filter to top-k tokens
  - Sample from nucleus
    ↓
Next Token: 1395 ("def")
    ↓
Repeat for each token:
  def → my → _function → ( → ) → : → \n → ...
    ↓
Complete Response: "def my_function():\n    pass"
    ↓
Add to History: [
  {role: "user", content: "Write a Python function"},
  {role: "assistant", content: "def my_function():\n    pass"}
]
    ↓
Display to User
```

---

## Troubleshooting Common Issues

### Issue 1: Out of Memory (OOM)
**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Causes:**
- Model too large for GPU VRAM
- Sequence length too long
- Batch size too large

**Solutions:**
1. Use smaller model
2. Reduce `max_length` in config
3. Clear CUDA cache: `torch.cuda.empty_cache()`
4. Use CPU instead of GPU (slower but more RAM)

---

### Issue 2: Slow Inference
**Symptoms:**
- Responses take 30+ seconds
- GPU utilization low

**Causes:**
- Running on CPU instead of GPU
- Model not fully loaded to GPU
- Other processes using GPU

**Solutions:**
1. Check device: `print(model.device)`
2. Check GPU usage: `nvidia-smi`
3. Close other GPU-heavy programs
4. Use `device_map="auto"` in model loading

---

### Issue 3: CUDA Errors
**Symptoms:**
```
CUDA error: no kernel image is available
```

**Causes:**
- PyTorch not compiled for your GPU architecture
- Driver/CUDA version mismatch

**Solutions:**
1. Update NVIDIA driver
2. Install correct PyTorch version
3. Use PyTorch nightly for new GPUs
4. See `docs/SM_120_FIX_PLAN.md`

---

### Issue 4: Model Not Found
**Symptoms:**
```
OSError: Can't load tokenizer for 'model-name'
```

**Causes:**
- Model name typo
- No internet connection
- Hugging Face Hub down
- Private model without token

**Solutions:**
1. Check model name spelling
2. Check internet connection
3. Provide Hugging Face token if needed
4. Check if model is cached: `ls model_cache/`

---

## Performance Optimization Tips

### 1. Use Half Precision (FP16)
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # ← Half precision
).to("cuda")
```
**Benefits:** 2x less memory, faster inference

### 2. Enable Flash Attention
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"  # ← Faster attention
).to("cuda")
```
**Benefits:** 2-4x faster attention computation

### 3. Use KV Cache
```python
# Automatically enabled in model.generate()
# Avoids recomputing attention keys/values
```
**Benefits:** Faster multi-turn conversations

### 4. Quantization (Advanced)
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # ← 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```
**Benefits:** 4x less memory, minimal quality loss

---

## Glossary

**Attention:** Mechanism for model to focus on relevant parts of input  
**Batch Size:** Number of inputs processed simultaneously  
**CUDA:** NVIDIA's GPU programming platform  
**Decoder:** Part of model that generates output tokens  
**Embedding:** Dense vector representation of a token  
**Encoder:** Part of model that processes input (not used in GPT-style models)  
**Fine-tuning:** Additional training on specific task  
**GPU:** Graphics Processing Unit (parallel processor)  
**Inference:** Using a trained model to make predictions  
**KV Cache:** Cached attention keys/values for efficiency  
**Logits:** Raw model outputs before softmax  
**Parameters:** Learned weights in the model  
**Pre-training:** Initial training on large text corpus  
**Quantization:** Reducing precision of weights (e.g., 32-bit → 4-bit)  
**Sampling:** Randomly selecting next token based on probabilities  
**Sequence Length:** Number of tokens in input/output  
**Softmax:** Converts logits to probabilities  
**Temperature:** Controls randomness in sampling  
**Tensor:** Multi-dimensional array  
**Tokenization:** Converting text to numbers  
**Top-k:** Only consider k most likely tokens  
**Top-p (Nucleus):** Consider tokens with cumulative probability p  
**Transformer:** Neural network architecture for sequence processing  
**VRAM:** Video RAM (GPU memory)  
**Weights:** Parameters (used interchangeably)

---

## Additional Resources

### Official Documentation
- **PyTorch:** https://pytorch.org/docs/stable/index.html
- **Transformers:** https://huggingface.co/docs/transformers/
- **CUDA:** https://docs.nvidia.com/cuda/

### Learning Resources
- **Transformer Architecture:** "Attention is All You Need" paper
- **LLM Fundamentals:** Andrej Karpathy's YouTube series
- **Hugging Face Course:** https://huggingface.co/learn/nlp-course/

### Troubleshooting
- **PyTorch Forums:** https://discuss.pytorch.org/
- **Hugging Face Forums:** https://discuss.huggingface.co/
- **Stack Overflow:** Tag: `pytorch`, `transformers`, `huggingface`

---

**Last Updated:** 2025-01-20  
**Version:** 1.0  
**Author:** GitHub Copilot

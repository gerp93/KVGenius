"""Test the chatbot with updated generation parameters."""
import fix_dll_paths
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.chat import ChatBot

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    'deepseek-ai/deepseek-coder-6.7b-instruct',
    cache_dir='./model_cache',
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    'deepseek-ai/deepseek-coder-6.7b-instruct',
    cache_dir='./model_cache',
    trust_remote_code=True
)

print(f"Model on: {model.device}")
print(f"GPU Memory: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB\n")

# Create chatbot
chatbot = ChatBot(model=model, tokenizer=tokenizer, device='cuda')

# Test conversation
test_messages = [
    "Hello! Can you help me?",
    "Write a simple Python function to add two numbers.",
]

for msg in test_messages:
    print(f"You: {msg}")
    try:
        response = chatbot.generate_response(msg, max_length=100)
        print(f"Bot: {response}\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
        import traceback
        traceback.print_exc()

print(f"Final GPU Memory: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

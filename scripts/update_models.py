#!/usr/bin/env python3
"""Update AVAILABLE_MODELS with detailed descriptions"""

# Read the file
with open('web_app_multi.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the new models dictionary
new_models = '''AVAILABLE_MODELS = {
    "Nous-Hermes-2-Mistral-7B": {
        "id": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "description": "🎭 Excellent roleplay & instruction-following",
        "details": "Fine-tuned on DPO (Direct Preference Optimization) for superior instruction following and character roleplay. Best all-around choice for conversational AI and creative scenarios.",
        "best_for": "Character roleplay, instruction following, general conversation",
        "params": "7B",
        "vram": "~14 GB"
    },
    "SynthIA-7B": {
        "id": "migtissera/SynthIA-7B-v2.0",
        "description": "🎪 Specialized for character roleplay",
        "details": "Trained specifically for character impersonation and creative storytelling. Excels at maintaining consistent personalities and engaging in immersive roleplay scenarios.",
        "best_for": "Character roleplay, creative writing, storytelling",
        "params": "7B",
        "vram": "~14 GB"
    },
    "Dark Champion MOE": {
        "id": "DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B",
        "description": "🔥 Uncensored creative writing (MOE)",
        "details": "Mixture-of-Experts model with 8 expert networks (3B each). Uncensored and abliterated for maximum creative freedom. Most intelligent option but requires more VRAM.",
        "best_for": "Creative writing, complex scenarios, unrestricted content",
        "params": "18.4B (8x3B)",
        "vram": "~16-20 GB"
    },
    "DeepSeek Coder 6.7B": {
        "id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "description": "💻 Code-focused model",
        "details": "Specialized for programming tasks, code generation, debugging, and technical explanations. Trained on massive code repositories with strong understanding of multiple languages.",
        "best_for": "Code generation, debugging, technical explanations, programming help",
        "params": "6.7B",
        "vram": "~13 GB"
    },
    "DeepSeek LLM 7B": {
        "id": "deepseek-ai/deepseek-llm-7b-chat",
        "description": "💬 General conversation",
        "details": "Balanced general-purpose model good for everyday conversations, Q&A, and informational queries. Reliable for factual responses and casual chat.",
        "best_for": "General conversation, Q&A, information retrieval",
        "params": "7B",
        "vram": "~14 GB"
    },
    "Wizard Vicuna 7B": {
        "id": "ehartford/Wizard-Vicuna-7B-Uncensored",
        "description": "🧙 Uncensored (not great for roleplay)",
        "details": "Older uncensored model. Less consistent for roleplay compared to newer options but completely unrestricted. Consider using Nous-Hermes-2 or Dark Champion instead.",
        "best_for": "Unrestricted content (older generation)",
        "params": "7B",
        "vram": "~14 GB"
    },
    "DialoGPT Medium": {
        "id": "microsoft/DialoGPT-medium",
        "description": "⚡ Fast & lightweight",
        "details": "Smallest and fastest option. Good for quick responses and casual chat when VRAM is limited. Less capable for complex tasks or roleplay.",
        "best_for": "Quick responses, low VRAM usage, casual chat",
        "params": "355M",
        "vram": "~2 GB"
    },
}'''

# Find and replace the AVAILABLE_MODELS section
import re
pattern = r'AVAILABLE_MODELS = \{[^}]*(?:\{[^}]*\}[^}]*)*\}'
content = re.sub(pattern, new_models, content, count=1, flags=re.DOTALL)

# Write back
with open('web_app_multi.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Updated AVAILABLE_MODELS with detailed descriptions!")

"""
Multi-Model Web interface for the AI chatbot using Gradio.
"""
# CRITICAL: Import DLL path fix BEFORE any torch imports
import fix_dll_paths

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ModelLoader
from src.chat import ChatBot
from src.utils import load_config, setup_logging, load_environment
import gradio as gr
import logging
import torch

logger = logging.getLogger(__name__)

# Global variables
active_models = {}
gen_config = {}
cache_dir = "./model_cache"
is_downloading = False
download_canceled = False

# Available models
AVAILABLE_MODELS = {
    "DeepSeek Coder 6.7B": {
        "id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "description": "💻 Code-focused model",
        "params": "6.7B",
        "vram": "~13 GB"
    },
    "DeepSeek LLM 7B": {
        "id": "deepseek-ai/deepseek-llm-7b-chat",
        "description": "💬 General conversation",
        "params": "7B",
        "vram": "~14 GB"
    },
    "DialoGPT Medium": {
        "id": "microsoft/DialoGPT-medium",
        "description": "⚡ Fast & lightweight",
        "params": "355M",
        "vram": "~2 GB"
    },
    "Dark Champion MOE": {
        "id": "DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B",
        "description": "🔥 Uncensored creative writing (MOE)",
        "params": "18.4B (8x3B)",
        "vram": "~16-20 GB"
    },
    "Wizard Vicuna 7B": {
        "id": "ehartford/Wizard-Vicuna-7B-Uncensored",
        "description": "🧙 Uncensored Wizard-Vicuna",
        "params": "7B",
        "vram": "~14 GB"
    },
}


def load_model(model_key, progress=gr.Progress()):
    """Load a model if not cached."""
    global is_downloading, download_canceled
    
    if model_key in active_models:
        return active_models[model_key]
    
    model_info = AVAILABLE_MODELS[model_key]
    model_id = model_info["id"]
    
    logger.info(f"Loading {model_key}...")
    
    # Check if model is cached
    model_path = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
    is_cached = os.path.exists(model_path)
    
    if not is_cached:
        is_downloading = True
        download_canceled = False
        progress(0, desc=f"📥 Downloading {model_key}...")
    else:
        progress(0, desc=f"⏳ Loading {model_key} from cache...")
    
    try:
        model_loader = ModelLoader(
            model_name=model_id,
            cache_dir=cache_dir,
            device="auto",
            token=None
        )
        
        # Simulate progress updates (Hugging Face downloads happen in background)
        if not is_cached:
            for i in range(5):
                if download_canceled:
                    is_downloading = False
                    raise Exception("Download canceled by user")
                progress((i + 1) * 0.2, desc=f"📥 Downloading {model_key}... {(i+1)*20}%")
                import time
                time.sleep(0.5)
        
        model, tokenizer = model_loader.load()
        
        chatbot = ChatBot(
            model=model,
            tokenizer=tokenizer,
            device=model_loader.device,
            max_history=5
        )
        
        active_models[model_key] = (model, tokenizer, chatbot)
        logger.info(f"✓ {model_key} loaded")
        
        is_downloading = False
        progress(1.0, desc=f"✅ {model_key} ready!")
        
        return active_models[model_key]
    
    except Exception as e:
        is_downloading = False
        raise e


def get_status():
    """Get system status."""
    lines = []
    
    if is_downloading:
        lines.append("📥 **Status:** Downloading model...")
        lines.append("")
    
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        lines.append(f"🎮 **GPU:** {gpu}")
        lines.append(f"📊 **VRAM:** {vram_used:.2f} / {vram_total:.1f} GB")
    else:
        lines.append("💻 **Device:** CPU")
    
    if active_models:
        lines.append(f"\n✅ **Loaded:** {len(active_models)} model(s)")
        for name in active_models.keys():
            lines.append(f"  • {name}")
    else:
        lines.append("\n📦 **No models loaded**")
    
    return "\n".join(lines)


def chat(message, history, model_selection):
    """Handle chat."""
    if not message.strip():
        return history
    
    try:
        model, tokenizer, chatbot = load_model(model_selection)
        
        # Keep only last 3 exchanges to prevent context confusion
        recent_history = history[-3:] if len(history) > 3 else history
        
        # Build a clean prompt with limited history
        prompt_parts = []
        
        # Add recent history
        for user_msg, bot_msg in recent_history:
            prompt_parts.append(f"User: {user_msg}")
            prompt_parts.append(f"Assistant: {bot_msg}")
        
        # Add current message
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        prompt = "\n".join(prompt_parts)
        
        logger.info(f"User message: {message}")
        logger.info(f"Using {len(recent_history)} previous exchanges")
        logger.info(f"Full prompt:\n{prompt}\n---")
        
        # Tokenize with proper limits
        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # Give more room for context
        ).to(chatbot.device)
        
        logger.info(f"Input tokens: {inputs.shape[1]}")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=150,  # Limit response length
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,  # Higher to prevent repetition
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # More aggressive response cleaning
        stop_phrases = ["\nUser:", "\nAssistant:", "User:", "Assistant:", "\n\n\n"]
        for stop in stop_phrases:
            if stop in response:
                response = response.split(stop)[0].strip()
        
        # Remove incomplete sentences at the end
        if response and not response[-1] in '.!?':
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        logger.info(f"Generated response: {response}\n---")
        
        history.append((message, response))
        return history
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        history.append((message, f"❌ Error: {str(e)}"))
        return history


def clear_chat(model_selection):
    """Clear history."""
    if model_selection in active_models:
        _, _, chatbot = active_models[model_selection]
        chatbot.reset_conversation()
    return []


def unload(model_selection):
    """Unload model."""
    if model_selection in active_models:
        del active_models[model_selection]
        torch.cuda.empty_cache()
        return get_status()
    return get_status()


def cancel_download():
    """Cancel ongoing download."""
    global download_canceled
    download_canceled = True
    return "🛑 Canceling download..."


def create_ui():
    """Create UI."""
    
    with gr.Blocks(title="KVGenius Multi-Model Chat", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# 🤖 KVGenius - Multi-Model AI Chat\n### RTX 5070 Ti (sm_120 Blackwell)")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(label="Chat", height=500, show_copy_button=True)
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type here and press Enter...",
                        lines=1,
                        scale=4
                    )
                    send = gr.Button("Send 📤", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear 🗑️", variant="secondary")
                    
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Model")
                
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=list(AVAILABLE_MODELS.keys())[0],
                    label="Select Model"
                )
                
                model_info = gr.Markdown()
                
                def update_info(model_key):
                    info = AVAILABLE_MODELS[model_key]
                    return f"""**{model_key}**

{info['description']}

📊 {info['params']} parameters
💾 {info['vram']} VRAM"""
                
                model_dropdown.change(update_info, model_dropdown, model_info)
                demo.load(lambda: update_info(list(AVAILABLE_MODELS.keys())[0]), outputs=model_info)
                
                gr.Markdown("---\n### 📊 Status")
                
                status = gr.Markdown()
                
                with gr.Row():
                    refresh = gr.Button("🔄 Refresh", size="sm")
                    cancel_btn = gr.Button("🛑 Cancel DL", size="sm", variant="stop")
                
                unload_btn = gr.Button("💾 Unload Model", size="sm")
                
                demo.load(get_status, outputs=status)
        
        with gr.Accordion("ℹ️ Help", open=False):
            gr.Markdown("""
**Features:**
- 🔄 Switch models anytime
- 💾 Models cached in VRAM
- 🗑️ Unload to free VRAM
- 📝 Copy any message

**Models:**
- **DeepSeek Coder** - Programming
- **DeepSeek LLM** - Chat
- **DialoGPT** - Fast/Small

**Privacy:**
- ✅ Runs locally
- ✅ No data sent out
- ✅ History in memory only
            """)
        
        # Events
        msg.submit(chat, [msg, chatbot_ui, model_dropdown], chatbot_ui).then(
            lambda: "", outputs=msg
        ).then(get_status, outputs=status)
        
        send.click(chat, [msg, chatbot_ui, model_dropdown], chatbot_ui).then(
            lambda: "", outputs=msg
        ).then(get_status, outputs=status)
        
        clear_btn.click(clear_chat, model_dropdown, chatbot_ui)
        refresh.click(get_status, outputs=status)
        unload_btn.click(unload, model_dropdown, status)
        cancel_btn.click(cancel_download, outputs=status).then(get_status, outputs=status)
    
    return demo


def main():
    """Main."""
    print("=" * 60)
    print("KVGenius Multi-Model Chat")
    print("=" * 60)
    
    load_environment()
    config = load_config()
    setup_logging(config.get('app', {}).get('log_level', 'INFO'))
    
    global gen_config, cache_dir
    gen_config = config.get('generation', {})
    cache_dir = config.get('model', {}).get('cache_dir', './model_cache')
    
    print(f"\n✓ Config loaded")
    print(f"✓ Cache: {cache_dir}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ PyTorch: {torch.__version__}")
    
    demo = create_ui()
    
    print("\n" + "=" * 60)
    print("✓ Starting server on http://127.0.0.1:7860")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()

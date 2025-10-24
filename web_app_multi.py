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
from src.database import ChatHistoryDB
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
db = None  # Database instance
current_conversation_id = None  # Active conversation ID

# Available models
AVAILABLE_MODELS = {
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
    }
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


def format_prompt_for_model(tokenizer, model_key, system_instruction, char_name, user_persona_context, recent_history, current_message):
    """
    Format prompt using model-specific templates.
    Tries tokenizer's chat template first, then falls back to model-specific formats.
    """
    
    # Build conversation messages
    messages = []
    
    # Add system message if present
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    
    # Add user persona context if present
    if user_persona_context:
        if messages and messages[-1]["role"] == "system":
            # Append to system message
            messages[-1]["content"] += f"\n\n{user_persona_context}"
        else:
            messages.append({"role": "system", "content": user_persona_context})
    
    # Add conversation history
    for user_msg, bot_msg in recent_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    # Add current message
    messages.append({"role": "user", "content": current_message})
    
    # DISABLED: Tokenizer chat templates are unreliable for multi-turn conversations
    # Always use our custom formats for better control
    # if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
    #     try:
    #         prompt = tokenizer.apply_chat_template(
    #             messages, 
    #             tokenize=False, 
    #             add_generation_prompt=True
    #         )
    #         logger.info(f"Using tokenizer chat template for {model_key}")
    #         return prompt
    #     except Exception as e:
    #         logger.warning(f"Chat template failed for {model_key}: {e}. Using fallback.")
    
    # Use model-specific formats for reliable multi-turn conversations
    logger.info(f"Using custom prompt format for {model_key}")
    
    # Mistral/Hermes format: [INST] ... [/INST]
    if "Mistral" in model_key or "Hermes" in model_key:
        prompt_parts = []
        
        # Add system instruction in its own [INST] block
        if system_instruction:
            prompt_parts.append(f"[INST] {system_instruction}")
            if user_persona_context:
                prompt_parts.append(f"\n{user_persona_context}")
            prompt_parts.append(" [/INST] Understood.</s>")  # Close system instruction
        
        # Add conversation history - each gets its own [INST] block
        for user_msg, bot_msg in recent_history:
            prompt_parts.append(f"[INST] {user_msg} [/INST] {bot_msg}</s>")
        
        # Add current message
        prompt_parts.append(f"[INST] {current_message} [/INST]")
        
        return "".join(prompt_parts)
    
    # Vicuna format: USER: ... ASSISTANT:
    elif "Vicuna" in model_key:
        prompt_parts = []
        if system_instruction:
            prompt_parts.append(system_instruction)
            if user_persona_context:
                prompt_parts.append(f"\n{user_persona_context}")
            prompt_parts.append("\n\n")
        
        for user_msg, bot_msg in recent_history:
            prompt_parts.append(f"USER: {user_msg}\nASSISTANT: {bot_msg}\n")
        
        prompt_parts.append(f"USER: {current_message}\nASSISTANT:")
        return "".join(prompt_parts)
    
    # Llama-3 format (used by Dark Champion MOE)
    elif "Llama" in model_key or "Champion" in model_key:
        prompt_parts = []
        if system_instruction:
            prompt_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_instruction}")
            if user_persona_context:
                prompt_parts.append(f"\n{user_persona_context}")
            prompt_parts.append("<|eot_id|>")
        
        for user_msg, bot_msg in recent_history:
            prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>")
            prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{bot_msg}<|eot_id|>")
        
        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{current_message}<|eot_id|>")
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(prompt_parts)
    
    # DeepSeek format (ChatML-like)
    elif "DeepSeek" in model_key:
        prompt_parts = []
        if system_instruction:
            prompt_parts.append(f"<｜begin▁of▁sentence｜>System: {system_instruction}")
            if user_persona_context:
                prompt_parts.append(f"\n{user_persona_context}")
            prompt_parts.append("\n")
        
        for user_msg, bot_msg in recent_history:
            prompt_parts.append(f"User: {user_msg}\nAssistant: {bot_msg}\n")
        
        prompt_parts.append(f"User: {current_message}\nAssistant:")
        return "".join(prompt_parts)
    
    # DialoGPT and generic fallback: simple format
    else:
        prompt_parts = []
        if system_instruction:
            prompt_parts.append(f"{system_instruction}\n\n")
            if user_persona_context:
                prompt_parts.append(f"{user_persona_context}\n\n")
        
        for user_msg, bot_msg in recent_history:
            prompt_parts.append(f"User: {user_msg}\nAssistant: {bot_msg}\n")
        
        prompt_parts.append(f"User: {current_message}\nAssistant:")
        return "".join(prompt_parts)


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


def chat(message, history, model_selection, ai_character_selection, user_persona_selection):
    """Handle chat."""
    global current_conversation_id, db
    
    if not message.strip():
        return history
    
    try:
        # Create new conversation if needed
        if current_conversation_id is None:
            ai_char_id = None
            user_persona_id = None
            
            if ai_character_selection != "None":
                ai_characters = db.get_all_ai_characters()
                for char in ai_characters:
                    if char['name'] == ai_character_selection:
                        ai_char_id = char['id']
                        break
            
            if user_persona_selection != "None":
                user_personas = db.get_all_user_personas()
                for persona in user_personas:
                    if persona['name'] == user_persona_selection:
                        user_persona_id = persona['id']
                        break
            
            current_conversation_id = db.create_conversation(
                model=model_selection,
                ai_character_id=ai_char_id,
                user_persona_id=user_persona_id
            )
            logger.info(f"Started new conversation {current_conversation_id}")
        
        model, tokenizer, chatbot = load_model(model_selection)
        
        # Keep only last 3 exchanges to prevent context confusion
        recent_history = history[-3:] if len(history) > 3 else history
        
        # Get AI character details
        system_instruction = None
        char_name = None
        if ai_character_selection != "None":
            ai_characters = db.get_all_ai_characters()
            for char in ai_characters:
                if char['name'] == ai_character_selection:
                    system_instruction = char.get('system_prompt')
                    char_name = char['name']
                    break
        
        # Get user persona context
        user_persona_context = None
        if user_persona_selection != "None":
            user_personas = db.get_all_user_personas()
            for persona in user_personas:
                if persona['name'] == user_persona_selection:
                    context_parts = [f"The user is roleplaying as {persona['name']}."]
                    if persona['description']:
                        context_parts.append(persona['description'])
                    if persona['background']:
                        context_parts.append(persona['background'])
                    user_persona_context = " ".join(context_parts)
                    break
        
        # Format prompt using model-specific template
        prompt = format_prompt_for_model(
            tokenizer=tokenizer,
            model_key=model_selection,
            system_instruction=system_instruction,
            char_name=char_name,
            user_persona_context=user_persona_context,
            recent_history=recent_history,
            current_message=message
        )
        
        logger.info(f"User message: {message}")
        logger.info(f"Model: {model_selection}")
        logger.info(f"Using {len(recent_history)} previous exchanges")
        logger.info(f"AI Character: {ai_character_selection}")
        logger.info(f"System Instruction Length: {len(system_instruction) if system_instruction else 0} chars")
        logger.info(f"Full prompt:\n{prompt}\n---")
        
        # Tokenize with proper limits
        # Use higher limit for character roleplay (long system prompts)
        # truncation_side='left' keeps the most recent conversation (at the end)
        tokenizer.truncation_side = 'left'
        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536  # Much higher limit for character roleplay
        ).to(chatbot.device)
        
        logger.info(f"Input tokens: {inputs.shape[1]}")
        
        # Get generation parameters from AI character if selected
        temperature = 0.7
        top_p = 0.9
        top_k = 50
        
        if ai_character_selection != "None":
            ai_characters = db.get_all_ai_characters()
            for char in ai_characters:
                if char['name'] == ai_character_selection:
                    temperature = char.get('temperature', 0.7)
                    top_p = char.get('top_p', 0.9)
                    top_k = char.get('top_k', 50)
                    break
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=150,  # Limit response length
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
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
        
        # Model-specific response cleaning
        stop_phrases = []
        
        # Common stop phrases for all models
        stop_phrases.extend(["\nUser:", "\nAssistant:", "User:", "Assistant:"])
        
        # Model-specific stop phrases
        if "Mistral" in model_selection or "Hermes" in model_selection:
            stop_phrases.extend(["[INST]", "[/INST]", "</s>"])
        elif "Vicuna" in model_selection:
            stop_phrases.extend(["USER:", "ASSISTANT:"])
        elif "Llama" in model_selection or "Champion" in model_selection:
            stop_phrases.extend(["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"])
        elif "DeepSeek" in model_selection:
            stop_phrases.extend(["<｜end▁of▁sentence｜>"])
        
        # Add character name stop phrases
        if char_name:
            stop_phrases.extend([f"\n{char_name}:", f"{char_name}:"])
        
        # Clean up response
        for stop in stop_phrases:
            if stop in response:
                response = response.split(stop)[0].strip()
        
        # Remove multiple newlines
        while "\n\n\n" in response:
            response = response.replace("\n\n\n", "\n\n")
        
        # Remove incomplete sentences at the end (optional, can be disabled)
        if response and len(response) > 20 and not response[-1] in '.!?"\')':
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        logger.info(f"Generated response: {response}\n---")
        
        # Save to database
        db.add_message(current_conversation_id, "user", message)
        db.add_message(current_conversation_id, "assistant", response)
        
        history.append((message, response))
        return history
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        history.append((message, f"❌ Error: {str(e)}"))
        return history


def clear_chat(model_selection):
    """Clear history and start new conversation."""
    global current_conversation_id
    current_conversation_id = None
    return []


def new_chat():
    """Start a new chat."""
    global current_conversation_id
    current_conversation_id = None
    return [], ""


def load_conversation(conv_title):
    """Load a conversation from database."""
    global current_conversation_id, db
    
    if not conv_title:
        return [], ""
    
    # Find conversation ID from title
    recent_convs = db.get_recent_conversations(20)
    for c in recent_convs:
        if f"{c['title']} ({c['model']})" == conv_title:
            current_conversation_id = c['id']
            history = db.get_conversation_messages(c['id'])
            return history, f"✅ Loaded: {c['title']}"
    
    return [], "❌ Conversation not found"


def export_conversation():
    """Export current conversation as JSON file for download."""
    global current_conversation_id, db
    
    if current_conversation_id is None:
        return None, "❌ No active conversation to export"
    
    try:
        import json
        import tempfile
        import os
        
        data = db.export_conversation(current_conversation_id)
        
        # Create filename
        safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in data['title'])
        filename = f"chat_{current_conversation_id}_{safe_title}.json"
        
        # Save to temp file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath, f"✅ Downloading {filename}"
    except Exception as e:
        return None, f"❌ Export failed: {str(e)}"


def delete_conversation():
    """Delete current conversation."""
    global current_conversation_id, db
    
    if current_conversation_id is None:
        return [], "❌ No active conversation to delete"
    
    try:
        db.delete_conversation(current_conversation_id)
        current_conversation_id = None
        return [], "✅ Conversation deleted"
    except Exception as e:
        return [], f"❌ Delete failed: {str(e)}"


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


def create_ai_character(name, system_prompt, description, temperature, top_p, top_k, avatar):
    """Create a new AI character."""
    global db
    
    if not name.strip():
        return {
            "status": "❌ Name is required",
            "char_list": list_ai_characters(),
            "name": name,
            "system": system_prompt,
            "desc": description,
            "temp": temperature,
            "topp": top_p,
            "topk": top_k,
            "avatar": avatar
        }
    
    try:
        db.create_ai_character(
            name=name,
            system_prompt=system_prompt,
            description=description,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            avatar=avatar
        )
        return {
            "status": f"✅ Created AI character: {name}",
            "char_list": list_ai_characters(),
            "name": "",
            "system": "",
            "desc": "",
            "temp": 0.7,
            "topp": 0.95,
            "topk": 50,
            "avatar": "🤖"
        }
    except Exception as e:
        return {
            "status": f"❌ Error: {str(e)}",
            "char_list": list_ai_characters(),
            "name": name,
            "system": system_prompt,
            "desc": description,
            "temp": temperature,
            "topp": top_p,
            "topk": top_k,
            "avatar": avatar
        }


def create_user_persona(name, description, background, avatar):
    """Create a new user persona."""
    global db
    
    if not name.strip():
        return {
            "status": "❌ Name is required",
            "persona_list": list_user_personas(),
            "name": name,
            "desc": description,
            "bg": background,
            "avatar": avatar
        }
    
    try:
        db.create_user_persona(
            name=name,
            description=description,
            background=background,
            avatar=avatar
        )
        return {
            "status": f"✅ Created user persona: {name}",
            "persona_list": list_user_personas(),
            "name": "",
            "desc": "",
            "bg": "",
            "avatar": "👤"
        }
    except Exception as e:
        return {
            "status": f"❌ Error: {str(e)}",
            "persona_list": list_user_personas(),
            "name": name,
            "desc": description,
            "bg": background,
            "avatar": avatar
        }


def list_ai_characters():
    """List all AI characters."""
    global db
    
    characters = db.get_all_ai_characters()
    if not characters:
        return "No AI characters yet."
    
    output = []
    for char in characters:
        output.append(f"**{char['avatar']} {char['name']}**")
        if char['description']:
            output.append(f"_{char['description']}_")
        output.append(f"Temperature: {char['temperature']}, Top-P: {char['top_p']}, Top-K: {char['top_k']}")
        if char['system_prompt']:
            output.append(f"Prompt: {char['system_prompt'][:100]}...")
        output.append("---")
    
    return "\n".join(output)


def list_user_personas():
    """List all user personas."""
    global db
    
    personas = db.get_all_user_personas()
    if not personas:
        return "No user personas yet."
    
    output = []
    for persona in personas:
        output.append(f"**{persona['avatar']} {persona['name']}**")
        if persona['description']:
            output.append(f"_{persona['description']}_")
        if persona['background']:
            output.append(f"Background: {persona['background'][:100]}...")
        output.append("---")
    
    return "\n".join(output)


def create_ai_character_tiles():
    """Create tile/card layout for AI characters with clickable navigation."""
    global db
    
    characters = db.get_all_ai_characters()
    
    # Generate tile HTML with direct Gradio integration instructions
    tiles_html = """
    <div style="padding: 10px;">
        <p style="color: #aaa; font-size: 14px; margin-bottom: 15px;">
            💡 <strong>To edit a character:</strong> Select it from the dropdown in the "Edit Character" tab
        </p>
    """
    
    tiles_html += """
    <style>
        .character-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            padding: 10px;
        }
        .character-tile {
            border: 2px solid #444;
            border-radius: 12px;
            padding: 15px;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        }
        .character-tile:hover {
            border-color: #00a8ff;
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0, 168, 255, 0.3);
        }
        .character-tile.create-new {
            border: 2px dashed #666;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 180px;
            background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        }
        .character-tile.create-new:hover {
            border-color: #00ff00;
            box-shadow: 0 5px 20px rgba(0, 255, 0, 0.2);
        }
        .tile-avatar {
            font-size: 48px;
            text-align: center;
            margin-bottom: 10px;
        }
        .tile-name {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #fff;
            text-align: center;
        }
        .tile-desc {
            font-size: 14px;
            color: #aaa;
            margin-bottom: 10px;
            text-align: center;
            min-height: 40px;
        }
        .tile-params {
            font-size: 12px;
            color: #888;
            text-align: center;
            border-top: 1px solid #444;
            padding-top: 8px;
            margin-top: 8px;
        }
        .create-icon {
            font-size: 64px;
            color: #666;
            text-align: center;
        }
        .create-text {
            font-size: 18px;
            color: #888;
            text-align: center;
            margin-top: 10px;
        }
    </style>
    <div class="character-grid">
    """
    
    # Create New tile
    tiles_html += """
        <div class="character-tile create-new">
            <div>
                <div class="create-icon">➕</div>
                <div class="create-text">Create New Character</div>
                <div style="font-size: 12px; color: #666; margin-top: 8px;">Go to Edit Character tab →</div>
            </div>
        </div>
    """
    
    for char in characters:
        desc = char['description'] if char['description'] else "No description"
        tiles_html += f"""
        <div class="character-tile">
            <div class="tile-avatar">{char['avatar']}</div>
            <div class="tile-name">{char['name']}</div>
            <div class="tile-desc">{desc}</div>
            <div class="tile-params">
                🌡️ {char['temperature']} | 🎯 P:{char['top_p']} K:{char['top_k']}
            </div>
        </div>
        """
    
    tiles_html += """
    </div>
    </div>
    """
    
    return tiles_html


def create_user_persona_tiles():
    """Create tile/card layout for user personas."""
    global db
    
    personas = db.get_all_user_personas()
    
    tiles_html = """
    <div style="padding: 10px;">
        <p style="color: #aaa; font-size: 14px; margin-bottom: 15px;">
            💡 <strong>To edit a persona:</strong> Select it from the dropdown in the "Edit Persona" tab
        </p>
    """
    
    tiles_html += """
    <style>
        .persona-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            padding: 10px;
        }
        .persona-tile {
            border: 2px solid #444;
            border-radius: 12px;
            padding: 15px;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        }
        .persona-tile:hover {
            border-color: #ff6b6b;
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(255, 107, 107, 0.3);
        }
        .persona-tile.create-new {
            border: 2px dashed #666;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 180px;
            background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        }
        .persona-tile.create-new:hover {
            border-color: #00ff00;
            box-shadow: 0 5px 20px rgba(0, 255, 0, 0.2);
        }
    </style>
    <div class="persona-grid">
    """
    
    # Create New tile
    tiles_html += """
        <div class="persona-tile create-new">
            <div>
                <div class="create-icon">➕</div>
                <div class="create-text">Create New Persona</div>
                <div style="font-size: 12px; color: #666; margin-top: 8px;">Go to Edit Persona tab →</div>
            </div>
        </div>
    """
    
    for persona in personas:
        desc = persona['description'] if persona['description'] else "No description"
        bg_preview = persona['background'][:80] + "..." if persona['background'] and len(persona['background']) > 80 else (persona['background'] or "No background")
        tiles_html += f"""
        <div class="persona-tile">
            <div class="tile-avatar">{persona['avatar']}</div>
            <div class="tile-name">{persona['name']}</div>
            <div class="tile-desc">{desc}</div>
            <div class="tile-params" style="font-size: 11px;">{bg_preview}</div>
        </div>
        """
    
    tiles_html += """
    </div>
    </div>
    """
    
    return tiles_html



def refresh_dropdowns():
    """Refresh AI character and user persona dropdowns."""
    global db
    
    ai_characters = db.get_all_ai_characters()
    ai_char_names = ["None"] + [char['name'] for char in ai_characters]
    
    user_personas = db.get_all_user_personas()
    user_persona_names = ["None"] + [p['name'] for p in user_personas]
    
    return gr.update(choices=ai_char_names), gr.update(choices=user_persona_names)


def load_ai_character_for_edit(char_name):
    """Load AI character data into form fields for editing."""
    global db
    
    if char_name == "[Create New]":
        return ("", "", "", "", 0.7, 0.95, 50, "🤖")
    
    ai_characters = db.get_all_ai_characters()
    for char in ai_characters:
        if char['name'] == char_name:
            return (
                str(char['id']),                    # hidden ID field
                char['name'],                       # name
                char['system_prompt'] or "",        # system prompt
                char['description'] or "",          # description
                char['temperature'],                # temperature
                char['top_p'],                      # top_p
                char['top_k'],                      # top_k
                char['avatar'] or "🤖"             # avatar
            )
    
    return ("", "", "", "", 0.7, 0.95, 50, "🤖")


def load_user_persona_for_edit(persona_name):
    """Load user persona data into form fields for editing."""
    global db
    
    if persona_name == "[Create New]":
        return ("", "", "", "", "👤")
    
    user_personas = db.get_all_user_personas()
    for persona in user_personas:
        if persona['name'] == persona_name:
            return (
                str(persona['id']),                 # hidden ID field
                persona['name'],                    # name
                persona['description'] or "",       # description
                persona['background'] or "",        # background
                persona['avatar'] or "👤"          # avatar
            )
    
    return ("", "", "", "", "👤")


def save_ai_character(char_id, name, system_prompt, description, temperature, top_p, top_k, avatar):
    """Save AI character (create new or update existing)."""
    global db
    
    if not name.strip():
        return "❌ Name is required"
    
    try:
        if char_id and char_id.strip():
            # Update existing character
            db.update_ai_character(
                int(char_id),
                name=name,
                system_prompt=system_prompt,
                description=description,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                avatar=avatar
            )
            return f"✅ Updated AI character: {name}"
        else:
            # Create new character
            db.create_ai_character(
                name=name,
                system_prompt=system_prompt,
                description=description,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                avatar=avatar
            )
            return f"✅ Created AI character: {name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def save_user_persona(persona_id, name, description, background, avatar):
    """Save user persona (create new or update existing)."""
    global db
    
    if not name.strip():
        return "❌ Name is required"
    
    try:
        if persona_id and persona_id.strip():
            # Update existing persona
            db.update_user_persona(
                int(persona_id),
                name=name,
                description=description,
                background=background,
                avatar=avatar
            )
            return f"✅ Updated user persona: {name}"
        else:
            # Create new persona
            db.create_user_persona(
                name=name,
                description=description,
                background=background,
                avatar=avatar
            )
            return f"✅ Created user persona: {name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"
        return "❌ Name is required", list_user_personas()
    
    try:
        if persona_id and persona_id.strip():
            # Update existing persona
            db.update_user_persona(
                int(persona_id),
                name=name,
                description=description,
                background=background,
                avatar=avatar
            )
            return f"✅ Updated user persona: {name}", list_user_personas()
        else:
            # Create new persona
            db.create_user_persona(
                name=name,
                description=description,
                background=background,
                avatar=avatar
            )
            return f"✅ Created user persona: {name}", list_user_personas()
    except Exception as e:
        return f"❌ Error: {str(e)}", list_user_personas()


def delete_ai_character_by_name(char_id, char_name):
    """Delete AI character by ID."""
    global db
    
    if not char_id or not char_id.strip():
        return "❌ Please select a character to delete"
    
    try:
        db.delete_ai_character(int(char_id))
        return f"✅ Deleted AI character: {char_name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def delete_user_persona_by_name(persona_id, persona_name):
    """Delete user persona by ID."""
    global db
    
    if not persona_id or not persona_id.strip():
        return "❌ Please select a persona to delete"
    
    try:
        db.delete_user_persona(int(persona_id))
        return f"✅ Deleted user persona: {persona_name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def handle_create_ai_character(name, system_prompt, description, temperature, top_p, top_k, avatar):
    """Wrapper to handle AI character creation and return proper outputs."""
    result = create_ai_character(name, system_prompt, description, temperature, top_p, top_k, avatar)
    
    return (
        result["status"],           # status message
        result["char_list"],        # updated character list
        result["name"],             # cleared/original name
        result["system"],           # cleared/original system prompt
        result["desc"],             # cleared/original description
        result["temp"],             # reset temperature
        result["topp"],             # reset top_p
        result["topk"],             # reset top_k
        result["avatar"]            # reset avatar
    )


def handle_create_user_persona(name, description, background, avatar):
    """Wrapper to handle user persona creation and return proper outputs."""
    result = create_user_persona(name, description, background, avatar)
    
    return (
        result["status"],           # status message
        result["persona_list"],     # updated persona list
        result["name"],             # cleared/original name
        result["desc"],             # cleared/original description
        result["bg"],               # cleared/original background
        result["avatar"]            # reset avatar
    )


def create_ui():
    """Create UI."""
    global db
    
    # Get AI characters and user personas for dropdowns
    ai_characters = db.get_all_ai_characters()
    ai_char_names = ["None"] + [char['name'] for char in ai_characters]
    
    user_personas = db.get_all_user_personas()
    user_persona_names = ["None"] + [p['name'] for p in user_personas]
    
    # Get recent conversations
    recent_convs = db.get_recent_conversations(20)
    conv_choices = {f"{c['title']} ({c['model']})": c['id'] for c in recent_convs}
    
    with gr.Blocks(title="KVGenius Multi-Model Chat", theme=gr.themes.Soft(), css="""
        /* Chat area - larger height */
        #chatbot {
            height: 600px !important;
            overflow-y: auto !important;
        }
        
        /* Side panel scrollable */
        .side-panel {
            overflow-y: auto !important;
            max-height: 700px !important;
        }
        
        /* Tile-style radio buttons */
        .tile-radio label {
            display: grid !important;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)) !important;
            gap: 15px !important;
            padding: 10px !important;
        }
        
        .tile-radio input[type="radio"] {
            display: none !important;
        }
        
        .tile-radio label > span {
            border: 2px solid #444 !important;
            border-radius: 12px !important;
            padding: 20px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%) !important;
            text-align: center !important;
            font-size: 16px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 100px !important;
        }
        
        /* Style Create New tile with orange color - MUST come after general style */
        .tile-radio label:first-of-type > span {
            background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
            border-color: #ff6b35 !important;
        }
        
        .tile-radio label > span:hover {
            border-color: #00a8ff !important;
            transform: translateY(-5px) !important;
            box-shadow: 0 5px 20px rgba(0, 168, 255, 0.3) !important;
        }
        
        /* Keep Create New orange even on hover */
        .tile-radio label:first-of-type > span:hover {
            border-color: #ff8555 !important;
            box-shadow: 0 5px 20px rgba(255, 107, 53, 0.5) !important;
            background: linear-gradient(135deg, #ff8555 0%, #ffa03e 100%) !important;
        }
        
        .tile-radio input[type="radio"]:checked + span {
            border-color: #00ff00 !important;
            background: linear-gradient(135deg, #1e3e1e 0%, #2a4a2a 100%) !important;
            box-shadow: 0 5px 20px rgba(0, 255, 0, 0.4) !important;
        }
        
        /* Keep Create New orange even when selected */
        .tile-radio input[type="radio"]:first-of-type:checked + span,
        .tile-radio label:first-of-type input[type="radio"]:checked + span {
            border-color: #ff6b35 !important;
            background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
            box-shadow: 0 5px 20px rgba(255, 107, 53, 0.6) !important;
        }
        
        /* Style Create New buttons with distinctive color */
        .create-new-btn {
            background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
            color: white !important;
            border: 2px solid #ff6b35 !important;
            font-weight: bold !important;
        }
        
        .create-new-btn:hover {
            background: linear-gradient(135deg, #ff8555 0%, #ffa03e 100%) !important;
            border-color: #ff8555 !important;
            transform: scale(1.05) !important;
        }
    """) as demo:
        
        gr.Markdown("# 🤖 KVGenius - Multi-Model AI Chat\n### RTX 5070 Ti (sm_120 Blackwell)")
        
        with gr.Tabs() as main_tabs:
            with gr.TabItem("💬 Chat", id="chat_tab"):
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=list(AVAILABLE_MODELS.keys()),
                        value=list(AVAILABLE_MODELS.keys())[0],
                        label="🤖 Model",
                        scale=2
                    )
                    
                    ai_char_dropdown = gr.Dropdown(
                        choices=ai_char_names,
                        value="None",
                        label="🎭 AI Character (What AI acts as)",
                        scale=2
                    )
                    
                    user_persona_dropdown = gr.Dropdown(
                        choices=user_persona_names,
                        value="None",
                        label="👤 User Persona (Who you are)",
                        scale=2
                    )
                    
                    new_chat_btn = gr.Button("🆕 New Chat", scale=1)
                    load_conv_dropdown = gr.Dropdown(
                        choices=list(conv_choices.keys()),
                        label="📂 Load Conversation",
                        scale=2
                    )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot_ui = gr.Chatbot(label="Chat", show_copy_button=True, elem_id="chatbot", type="tuples")
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                label="Message",
                                placeholder="Type here and press Enter...",
                                lines=1,
                                scale=4
                            )
                            send = gr.Button("Send 📤", variant="primary", scale=1)
                        
                        clear_btn = gr.Button("Clear 🗑️", variant="secondary")
                        
                        with gr.Row():
                            export_btn = gr.Button("📥 Export Chat", size="sm")
                            delete_btn = gr.Button("🗑️ Delete Chat", size="sm", variant="stop")
                        
                        # Hidden file output for export downloads
                        export_file = gr.File(label="Download", visible=False)
                            
                    with gr.Column(scale=1, elem_classes="side-panel"):
                        gr.Markdown("### 🎛️ Model")
                        
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
- 🎭 AI Characters control how the AI behaves
- 👤 User Personas define who you're roleplaying as

**Models (Roleplay Recommended ⭐):**
- **⭐ Nous-Hermes-2-Mistral** - Best instruction-following & roleplay
- **⭐ SynthIA-7B** - Specialized for character roleplay
- **⭐ Dark Champion MOE** - Most intelligent, creative writing
- **DeepSeek Coder** - Programming tasks
- **DeepSeek LLM** - General chat
- **Wizard Vicuna** - Uncensored (older, less consistent)
- **DialoGPT** - Fast/Small

**Privacy:**
- ✅ Runs locally
- ✅ No data sent out
- ✅ History saved in database
                    """)
            
            with gr.TabItem("🎭 Characters", id="char_tab"):
                # Browse view
                with gr.Column(visible=True) as char_browse_view:
                    gr.Markdown("## 🎴 Your AI Characters\nClick any character tile to edit it.")
                    
                    # Get character names for radio buttons
                    char_display_names = ["➕ Create New"] + [f"{char['avatar']} {char['name']}" for char in db.get_all_ai_characters()]
                    char_actual_names = ["[Create New]"] + [char['name'] for char in db.get_all_ai_characters()]
                    
                    ai_char_selector = gr.Radio(
                        choices=char_display_names,
                        label="Select Character",
                        elem_classes="tile-radio",
                        container=False
                    )
                    
                    refresh_ai_tiles_btn = gr.Button("🔄 Refresh Gallery", size="sm")
                
                # Edit view (hidden by default)
                with gr.Column(visible=False) as char_edit_view:
                    gr.Markdown("## Edit AI Character\nModify character personality, system prompt, and generation parameters.")
                    
                    back_to_browse_ai_btn = gr.Button("← Back to Gallery", size="sm", variant="secondary")
                    
                    ai_char_id_hidden = gr.Textbox(visible=False, value="")
                    
                    new_ai_name = gr.Textbox(label="Character Name", placeholder="e.g., Friendly Tutor")
                    new_ai_desc = gr.Textbox(label="Short Description", placeholder="Brief description for the dropdown")
                    new_ai_system = gr.Textbox(
                        label="System Prompt (Personality)", 
                        placeholder="You are a helpful tutor who explains things clearly...", 
                        lines=8
                    )
                    
                    gr.Markdown("### Generation Parameters")
                    with gr.Row():
                        new_ai_temp = gr.Slider(0.1, 1.5, value=0.7, label="Temperature", info="Creativity (higher = more random)")
                        new_ai_topp = gr.Slider(0.1, 1.0, value=0.95, label="Top P", info="Nucleus sampling")
                        new_ai_topk = gr.Slider(1, 100, value=50, step=1, label="Top K", info="Token diversity")
                    
                    new_ai_avatar = gr.Textbox(label="Avatar Emoji", value="🤖", max_lines=1)
                    
                    with gr.Row():
                        save_ai_btn = gr.Button("💾 Save Character", variant="primary", size="lg")
                        delete_ai_btn = gr.Button("🗑️ Delete Character", variant="stop", size="lg")
                    
                    ai_create_status = gr.Markdown()
            
            with gr.TabItem("👤 Personas", id="persona_tab"):
                # Browse view
                with gr.Column(visible=True) as persona_browse_view:
                    gr.Markdown("## 🎴 Your User Personas\nClick any persona tile to edit it.")
                    
                    # Get persona names for radio buttons
                    persona_display_names = ["➕ Create New"] + [f"{p['avatar']} {p['name']}" for p in db.get_all_user_personas()]
                    persona_actual_names = ["[Create New]"] + [p['name'] for p in db.get_all_user_personas()]
                    
                    user_persona_selector = gr.Radio(
                        choices=persona_display_names,
                        label="Select Persona",
                        elem_classes="tile-radio",
                        container=False
                    )
                    
                    refresh_user_tiles_btn = gr.Button("🔄 Refresh Gallery", size="sm")
                
                # Edit view (hidden by default)
                with gr.Column(visible=False) as persona_edit_view:
                    gr.Markdown("## Edit User Persona\nModify who you are in the roleplay.")
                    
                    back_to_browse_user_btn = gr.Button("← Back to Gallery", size="sm", variant="secondary")
                    
                    user_persona_id_hidden = gr.Textbox(visible=False, value="")
                    
                    new_user_name = gr.Textbox(label="Persona Name", placeholder="e.g., Space Explorer")
                    new_user_desc = gr.Textbox(label="Short Description", placeholder="Brief description (e.g., 'A brave starship captain')")
                    new_user_bg = gr.Textbox(
                        label="Background/Context", 
                        placeholder="Detailed roleplay background: You are Captain Sarah Chen, commanding the starship Odyssey...", 
                        lines=8
                    )
                    new_user_avatar = gr.Textbox(label="Avatar Emoji", value="👤", max_lines=1)
                    
                    with gr.Row():
                        save_user_btn = gr.Button("💾 Save Persona", variant="primary", size="lg")
                        delete_user_btn = gr.Button("🗑️ Delete Persona", variant="stop", size="lg")
                    
                    user_create_status = gr.Markdown()
        
        # Chat Events
        msg.submit(chat, [msg, chatbot_ui, model_dropdown, ai_char_dropdown, user_persona_dropdown], chatbot_ui).then(
            lambda: "", outputs=msg
        ).then(get_status, outputs=status)
        
        send.click(chat, [msg, chatbot_ui, model_dropdown, ai_char_dropdown, user_persona_dropdown], chatbot_ui).then(
            lambda: "", outputs=msg
        ).then(get_status, outputs=status)
        
        clear_btn.click(clear_chat, model_dropdown, chatbot_ui)
        new_chat_btn.click(new_chat, outputs=[chatbot_ui, status])
        
        load_conv_dropdown.change(load_conversation, load_conv_dropdown, [chatbot_ui, status])
        
        export_btn.click(export_conversation, outputs=[export_file, status])
        delete_btn.click(delete_conversation, outputs=[chatbot_ui, status])
        
        refresh.click(get_status, outputs=status)
        unload_btn.click(unload, model_dropdown, status)
        cancel_btn.click(cancel_download, outputs=status).then(get_status, outputs=status)
        
        # Define helper functions for character/persona management
        def refresh_char_selector():
            chars = db.get_all_ai_characters()
            display_names = ["➕ Create New"] + [f"{char['avatar']} {char['name']}" for char in chars]
            return gr.update(choices=display_names)
        
        def refresh_persona_selector():
            personas = db.get_all_user_personas()
            display_names = ["➕ Create New"] + [f"{p['avatar']} {p['name']}" for p in personas]
            return gr.update(choices=display_names)
        
        def open_char_editor(display_name):
            if display_name is None:
                return gr.update(visible=True), gr.update(visible=False), "", "", "", "", 0.7, 0.95, 50, "🤖"
            # Extract actual name from display format "🤖 Name"
            if display_name.startswith("➕"):
                # Return empty form for new character
                return gr.update(visible=False), gr.update(visible=True), "", "", "", "", 0.7, 0.95, 50, "🤖"
            else:
                actual_name = display_name.split(" ", 1)[1] if " " in display_name else display_name
                # Load existing character data
                ai_characters = db.get_all_ai_characters()
                for char in ai_characters:
                    if char['name'] == actual_name:
                        return (
                            gr.update(visible=False),  # hide browse
                            gr.update(visible=True),  # show edit
                            str(char['id']),  # hidden id
                            char['name'],
                            char['system_prompt'] or "",
                            char['description'] or "",
                            char['temperature'],
                            char['top_p'],
                            char['top_k'],
                            char['avatar'] or "🤖"
                        )
                # If not found, return empty
                return gr.update(visible=False), gr.update(visible=True), "", actual_name, "", "", 0.7, 0.95, 50, "🤖"
        
        def open_persona_editor(display_name):
            if display_name is None:
                return gr.update(visible=True), gr.update(visible=False), "", "", "", "", "👤"
            # Extract actual name from display format "👤 Name"
            if display_name.startswith("➕"):
                # Return empty form for new persona
                return gr.update(visible=False), gr.update(visible=True), "", "", "", "", "👤"
            else:
                actual_name = display_name.split(" ", 1)[1] if " " in display_name else display_name
                # Load existing persona data
                personas = db.get_all_user_personas()
                for persona in personas:
                    if persona['name'] == actual_name:
                        return (
                            gr.update(visible=False),  # hide browse
                            gr.update(visible=True),  # show edit
                            str(persona['id']),  # hidden id
                            persona['name'],
                            persona['description'] or "",
                            persona['background'] or "",
                            persona['avatar'] or "👤"
                        )
                # If not found, return empty
                return gr.update(visible=False), gr.update(visible=True), "", actual_name, "", "", "👤"
        
        # AI Character Management Events
        # Save (create or update) character
        save_ai_btn.click(
            save_ai_character,
            [ai_char_id_hidden, new_ai_name, new_ai_system, new_ai_desc, new_ai_temp, new_ai_topp, new_ai_topk, new_ai_avatar],
            [ai_create_status]
        ).then(
            refresh_dropdowns,
            outputs=[ai_char_dropdown, user_persona_dropdown]
        ).then(
            refresh_char_selector,
            outputs=ai_char_selector
        )
        
        # Delete character
        delete_ai_btn.click(
            delete_ai_character_by_name,
            [ai_char_id_hidden, new_ai_name],
            [ai_create_status]
        ).then(
            refresh_dropdowns,
            outputs=[ai_char_dropdown, user_persona_dropdown]
        ).then(
            lambda: ("", "", "", "", 0.7, 0.95, 50, "🤖"),
            outputs=[ai_char_id_hidden, new_ai_name, new_ai_system, new_ai_desc, new_ai_temp, new_ai_topp, new_ai_topk, new_ai_avatar]
        ).then(
            refresh_char_selector,
            outputs=ai_char_selector
        )
        
        # Tab navigation buttons for characters
        back_to_browse_ai_btn.click(
            lambda: (gr.update(visible=True), gr.update(visible=False), None),
            outputs=[char_browse_view, char_edit_view, ai_char_selector]
        )
        
        # Auto-navigate on tile selection
        ai_char_selector.change(
            open_char_editor,
            inputs=ai_char_selector,
            outputs=[char_browse_view, char_edit_view, ai_char_id_hidden, new_ai_name, new_ai_system, new_ai_desc, new_ai_temp, new_ai_topp, new_ai_topk, new_ai_avatar]
        )
        
        # Refresh character selector
        refresh_ai_tiles_btn.click(refresh_char_selector, outputs=ai_char_selector)
        
        # User Persona Management Events
        # Save (create or update) persona
        save_user_btn.click(
            save_user_persona,
            [user_persona_id_hidden, new_user_name, new_user_desc, new_user_bg, new_user_avatar],
            [user_create_status]
        ).then(
            refresh_dropdowns,
            outputs=[ai_char_dropdown, user_persona_dropdown]
        ).then(
            refresh_persona_selector,
            outputs=user_persona_selector
        )
        
        # Delete persona
        delete_user_btn.click(
            delete_user_persona_by_name,
            [user_persona_id_hidden, new_user_name],
            [user_create_status]
        ).then(
            refresh_dropdowns,
            outputs=[ai_char_dropdown, user_persona_dropdown]
        ).then(
            lambda: ("", "", "", "", "👤"),
            outputs=[user_persona_id_hidden, new_user_name, new_user_desc, new_user_bg, new_user_avatar]
        ).then(
            refresh_persona_selector,
            outputs=user_persona_selector
        )
        
        # Tab navigation buttons for personas
        back_to_browse_user_btn.click(
            lambda: (gr.update(visible=True), gr.update(visible=False), None),
            outputs=[persona_browse_view, persona_edit_view, user_persona_selector]
        )
        
        # Auto-navigate on tile selection
        user_persona_selector.change(
            open_persona_editor,
            inputs=user_persona_selector,
            outputs=[persona_browse_view, persona_edit_view, user_persona_id_hidden, new_user_name, new_user_desc, new_user_bg, new_user_avatar]
        )
        
        # Refresh persona selector
        refresh_user_tiles_btn.click(refresh_persona_selector, outputs=user_persona_selector)
    
    return demo


def main():
    """Main."""
    global gen_config, cache_dir, db
    
    print("=" * 60)
    print("KVGenius Multi-Model Chat")
    print("=" * 60)
    
    load_environment()
    config = load_config()
    setup_logging(config.get('app', {}).get('log_level', 'INFO'))
    
    gen_config = config.get('generation', {})
    cache_dir = config.get('model', {}).get('cache_dir', './model_cache')
    
    # Initialize database
    db = ChatHistoryDB()
    db.init_defaults()
    
    print(f"\n✓ Config loaded")
    print(f"✓ Cache: {cache_dir}")
    print(f"✓ Database: chat_history.db")
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ PyTorch: {torch.__version__}")
    
    # Pre-load your most-used model to avoid loading delay on first chat
    default_model = "Nous-Hermes-2-Mistral-7B"  # Best for roleplay
    print(f"\n⏳ Pre-loading {default_model}...")
    try:
        load_model(default_model)
        print(f"✓ {default_model} loaded and ready")
    except Exception as e:
        print(f"⚠ Could not pre-load model: {e}")
        print("  (Will load on first use instead)")
    
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

"""
Prompt enhancement for image generation.
Supports both template-based and AI-powered enhancement.
"""
import re
import logging
import torch

logger = logging.getLogger(__name__)

# CLIP token limit - applies to all SD/SDXL models (hardcoded in CLIP architecture)
CLIP_MAX_TOKENS = 77


def count_clip_tokens(text, tokenizer=None):
    """Count actual CLIP tokens using the model's tokenizer, or estimate if not available.
    
    Args:
        text: The text to count tokens for
        tokenizer: Optional CLIP tokenizer for accurate counting
    
    Returns:
        Estimated or actual token count
    """
    if not text:
        return 0
    
    # Try to use the actual tokenizer from loaded model
    if tokenizer is not None:
        try:
            tokens = tokenizer(text, return_tensors="pt", truncation=False)
            return tokens.input_ids.shape[1]
        except:
            pass
    
    # Fallback: estimate based on CLIP tokenization patterns
    # CLIP uses BPE - roughly 1.3 tokens per word on average
    words = len(re.findall(r'\w+', text))
    punctuation = len(re.findall(r'[^\w\s]', text))
    # Add buffer for subword tokenization
    return int(words * 1.3) + punctuation


def format_token_count(prompt, tokenizer=None):
    """Format token count for display with warnings.
    
    Args:
        prompt: The prompt text
        tokenizer: Optional CLIP tokenizer
    
    Returns:
        Formatted string with token count and warnings
    """
    if not prompt or not prompt.strip():
        return "*0 / 77 tokens*"
    
    tokens = count_clip_tokens(prompt, tokenizer)
    
    if tokens > CLIP_MAX_TOKENS:
        return f"### ⚠️ {tokens} / 77 TOKENS - PROMPT WILL BE TRUNCATED!"
    elif tokens > 60:
        return f"**{tokens} / 77 tokens** ⚠️ *getting close to limit*"
    else:
        return f"*{tokens} / 77 tokens*"


class PromptEnhancer:
    """Handles prompt enhancement using templates and AI models."""
    
    # Enhancement templates for different styles
    TEMPLATES = {
        "photorealistic": {
            "prefix": "",
            "suffix": ", RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
            "quality": ", masterpiece, best quality, highly detailed, sharp focus"
        },
        "artistic": {
            "prefix": "",
            "suffix": ", digital art, concept art, smooth, sharp focus",
            "quality": ", masterpiece, best quality, highly detailed, dramatic lighting, vibrant colors"
        },
        "portrait": {
            "prefix": "portrait of ",
            "suffix": ", professional portrait photography, studio lighting, 85mm lens, bokeh",
            "quality": ", masterpiece, detailed skin texture, sharp eyes, natural lighting"
        },
        "fantasy": {
            "prefix": "",
            "suffix": ", fantasy art, epic scene, magical atmosphere, ethereal lighting",
            "quality": ", masterpiece, best quality, highly detailed, dramatic composition"
        },
        "anime": {
            "prefix": "",
            "suffix": ", anime style, cel shading, vibrant colors",
            "quality": ", masterpiece, best quality, highly detailed, sharp lines"
        }
    }
    
    def __init__(self, cache_dir="./data/model_cache"):
        """Initialize the prompt enhancer.
        
        Args:
            cache_dir: Directory for caching AI models
        """
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
    
    def load_ai_model(self):
        """Lazy load a small T5 model for AI enhancement on CPU.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if self.model is not None:
            return True
        
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            logger.info("Loading prompt enhancer model on CPU...")
            model_name = "google/flan-t5-small"  # Small model, ~300MB, runs well on CPU
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=self.cache_dir
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            # Keep on CPU explicitly
            self.model = self.model.to("cpu")
            self.model.eval()
            
            logger.info("Prompt enhancer loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load prompt enhancer: {e}")
            return False
    
    def detect_style(self, prompt):
        """Auto-detect style from prompt content.
        
        Args:
            prompt: The input prompt
        
        Returns:
            Detected style string
        """
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["anime", "manga", "cel shading", "cartoon"]):
            return "anime"
        elif any(word in prompt_lower for word in ["fantasy", "magical", "dragon", "wizard", "fairy", "ethereal"]):
            return "fantasy"
        elif any(word in prompt_lower for word in ["portrait", "headshot", "face", "person closeup"]):
            return "portrait"
        elif any(word in prompt_lower for word in ["painting", "artwork", "artistic", "illustration", "concept art"]):
            return "artistic"
        else:
            return "photorealistic"
    
    def enhance_with_template(self, prompt, style="photorealistic"):
        """Enhance prompt using predefined templates (no AI required).
        
        Args:
            prompt: The input prompt
            style: Style to apply (photorealistic, artistic, portrait, fantasy, anime)
        
        Returns:
            Tuple of (enhanced_prompt, status_message)
        """
        if not prompt or not prompt.strip():
            return prompt, "❌ Please enter a prompt first"
        
        template = self.TEMPLATES.get(style, self.TEMPLATES["photorealistic"])
        
        enhanced = prompt.strip()
        if template["prefix"] and not enhanced.lower().startswith(template["prefix"]):
            enhanced = template["prefix"] + enhanced
        # Only add suffix, skip the generic "quality" terms
        enhanced += template["suffix"]
        
        token_count = count_clip_tokens(enhanced, None)
        return enhanced, f"✨ Quick {style} style! ({token_count}/77 tokens)"
    
    def enhance_smart(self, prompt, style="photorealistic"):
        """Smart enhancement that adds contextually relevant details.
        
        Args:
            prompt: The input prompt
            style: Style to apply
        
        Returns:
            Tuple of (enhanced_prompt, status_message)
        """
        if not prompt or not prompt.strip():
            return prompt, "❌ Please enter a prompt first"
        
        prompt_lower = prompt.lower()
        additions = []
        
        # Context-aware additions based on prompt content
        if any(word in prompt_lower for word in ["woman", "man", "person", "girl", "boy", "portrait"]):
            additions.extend(["detailed face", "expressive eyes"])
        if any(word in prompt_lower for word in ["landscape", "mountain", "forest", "ocean", "city"]):
            additions.extend(["atmospheric perspective", "volumetric lighting"])
        if any(word in prompt_lower for word in ["night", "dark", "evening"]):
            additions.extend(["dramatic shadows", "rim lighting"])
        if any(word in prompt_lower for word in ["sunset", "sunrise", "golden"]):
            additions.extend(["golden hour", "warm tones"])
        
        # Style-specific additions
        style_additions = {
            "photorealistic": ["RAW photo", "8k uhd", "dslr", "natural lighting"],
            "artistic": ["digital painting", "artstation trending", "vibrant colors"],
            "portrait": ["studio lighting", "85mm lens", "shallow depth of field"],
            "fantasy": ["magical atmosphere", "ethereal glow", "epic composition"],
            "anime": ["anime style", "cel shading", "clean lines"]
        }
        additions.extend(style_additions.get(style, style_additions["photorealistic"])[:3])
        
        # Build enhanced prompt
        enhanced = prompt.strip()
        if additions:
            enhanced += ", " + ", ".join(additions[:5])  # Limit to avoid token overflow
        
        token_count = count_clip_tokens(enhanced, None)
        return enhanced, f"✨ Smart enhanced! ({token_count}/77 tokens)"
    
    def enhance_with_ai(self, prompt, style="photorealistic"):
        """Use AI to enhance a basic prompt into a more detailed one.
        
        Args:
            prompt: The input prompt
            style: Style to apply
        
        Returns:
            Tuple of (enhanced_prompt, status_message)
        """
        if not prompt or not prompt.strip():
            return prompt, "❌ Please enter a prompt first"
        
        # Try to load model if not loaded
        if self.model is None:
            if not self.load_ai_model():
                # Fallback to template-based enhancement
                return self.enhance_with_template(prompt, style)
        
        try:
            # Style-specific context for better AI responses
            style_hints = {
                "photorealistic": "realistic photograph, natural lighting, detailed textures",
                "artistic": "digital artwork, vibrant colors, creative composition",
                "portrait": "portrait photography, facial details, studio lighting",
                "fantasy": "fantasy art, magical elements, epic atmosphere",
                "anime": "anime illustration, cel shading, expressive style"
            }
            style_hint = style_hints.get(style, style_hints["photorealistic"])
            
            # Better instruction that asks for specific additions
            instruction = f"""Add specific visual details to this image prompt. Include: setting/environment, lighting conditions, colors, textures, camera angle or composition. Style: {style_hint}

Prompt: {prompt.strip()}

Enhanced prompt:"""
            
            inputs = self.tokenizer(
                instruction,
                return_tensors="pt",
                max_length=256,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=100,
                    num_beams=4,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            ai_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Check if AI gave a meaningful response (not just generic text)
            generic_responses = ["masterpiece", "best quality", "highly detailed", "sharp focus"]
            is_generic = all(term in ai_output.lower() for term in generic_responses[:2]) and len(ai_output) < len(prompt) + 50
            
            if is_generic or len(ai_output) < 10:
                # AI didn't help much, use template with smart expansion
                return self.enhance_smart(prompt, style)
            
            # Combine original prompt idea with AI expansion
            # If AI output seems like a complete rewrite, use it; otherwise combine
            if prompt.lower().split()[0] in ai_output.lower():
                enhanced = ai_output
            else:
                enhanced = f"{prompt.strip()}, {ai_output}"
            
            # Add minimal style suffix (not the heavy quality terms)
            template = self.TEMPLATES.get(style, self.TEMPLATES["photorealistic"])
            enhanced += template["suffix"]
            
            token_count = count_clip_tokens(enhanced, None)
            return enhanced, f"✨ AI Enhanced! ({token_count}/77 tokens)"
            
        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            # Fallback to smart template
            return self.enhance_smart(prompt, style)
    
    def quick_enhance(self, prompt):
        """Quick enhancement using templates only (fast, no model loading).
        
        Args:
            prompt: The input prompt
        
        Returns:
            Tuple of (enhanced_prompt, status_message)
        """
        style = self.detect_style(prompt)
        return self.enhance_with_template(prompt, style)
    
    def ai_enhance(self, prompt):
        """Full AI enhancement (loads model if needed).
        
        Args:
            prompt: The input prompt
        
        Returns:
            Tuple of (enhanced_prompt, status_message)
        """
        style = self.detect_style(prompt)
        return self.enhance_with_ai(prompt, style)

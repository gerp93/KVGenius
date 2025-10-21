"""
Chat interface for interacting with the AI model.
"""
from typing import List, Dict, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class ChatBot:
    """Main chatbot interface for conversation management."""
    
    def __init__(self, model, tokenizer, device: str = "cpu", max_history: int = 5):
        """
        Initialize the chatbot.
        
        Args:
            model: The loaded language model
            tokenizer: The model's tokenizer
            device: Device to run inference on
            max_history: Maximum number of conversation turns to remember
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        
    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]
    
    def build_prompt(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """
        Build a prompt from conversation history using the model's chat template.
        
        Args:
            user_input: The user's current input
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Formatted prompt string
        """
        # Build messages in chat format
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in self.conversation_history:
            messages.append(msg)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                pass  # Fall back to simple format
        
        # Fallback to simple format
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(msg["content"])
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def generate_response(
        self,
        user_input: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: The user's message
            max_length: Maximum length of generated response
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        try:
            # Build prompt with history
            prompt = self.build_prompt(user_input)
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate response with proper stopping
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=min(max_length, 256),  # Reduced cap to 256 tokens
                    temperature=max(temperature, 0.1),  # Ensure temperature > 0
                    top_k=top_k if top_k > 0 else 50,
                    top_p=min(max(top_p, 0.1), 1.0),  # Clamp between 0.1 and 1.0
                    repetition_penalty=max(repetition_penalty, 1.0),
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    **kwargs
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Clean up response - stop at common breaking points
            stop_phrases = ["\nUser:", "\nAssistant:", "\n\n\n", "```\n###"]
            for stop in stop_phrases:
                if stop in response:
                    response = response.split(stop)[0].strip()
            
            # Add to history
            self.add_to_history("user", user_input)
            self.add_to_history("assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history

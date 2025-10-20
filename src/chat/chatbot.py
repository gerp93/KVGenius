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
        Build a prompt from conversation history.
        
        Args:
            user_input: The user's current input
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(system_prompt)
        
        # Add conversation history
        for msg in self.conversation_history:
            if msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            else:
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        # Add current user input
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def generate_response(
        self,
        user_input: str,
        max_length: int = 1000,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
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
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
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

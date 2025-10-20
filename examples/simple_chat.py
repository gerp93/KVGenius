"""
Simple example demonstrating basic chatbot usage.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ModelLoader
from src.chat import ChatBot


def main():
    """Simple example of using the chatbot."""
    print("Loading chatbot...")
    
    # Initialize model loader
    model_loader = ModelLoader(
        model_name="microsoft/DialoGPT-small",  # Using small model for faster loading
        device="auto"
    )
    
    # Load model and tokenizer
    model, tokenizer = model_loader.load()
    print("Model loaded!")
    
    # Create chatbot instance
    chatbot = ChatBot(
        model=model,
        tokenizer=tokenizer,
        device=model_loader.device,
        max_history=3
    )
    
    # Example conversation
    questions = [
        "Hello! How are you?",
        "What can you help me with?",
        "Tell me a fun fact."
    ]
    
    print("\n" + "=" * 60)
    print("Example Conversation")
    print("=" * 60 + "\n")
    
    for question in questions:
        print(f"You: {question}")
        response = chatbot.generate_response(
            question,
            max_length=50,
            temperature=0.7
        )
        print(f"Bot: {response}\n")
    
    print("=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    main()

"""
Example demonstrating different model configurations.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ModelLoader
from src.chat import ChatBot


def test_model(model_name: str):
    """Test a specific model."""
    print(f"\nTesting model: {model_name}")
    print("-" * 60)
    
    try:
        # Load model
        model_loader = ModelLoader(model_name=model_name, device="auto")
        model, tokenizer = model_loader.load()
        
        # Create chatbot
        chatbot = ChatBot(
            model=model,
            tokenizer=tokenizer,
            device=model_loader.device
        )
        
        # Test with a question
        question = "What is artificial intelligence?"
        print(f"Question: {question}")
        
        response = chatbot.generate_response(question, max_length=80)
        print(f"Response: {response}\n")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}\n")
        return False


def main():
    """Test different models."""
    print("=" * 60)
    print("Testing Different Models")
    print("=" * 60)
    
    # List of models to test (you can add more)
    models = [
        "microsoft/DialoGPT-small",
        "gpt2",
        # Add more models as needed
    ]
    
    results = {}
    for model_name in models:
        results[model_name] = test_model(model_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for model_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{status}: {model_name}")


if __name__ == "__main__":
    main()

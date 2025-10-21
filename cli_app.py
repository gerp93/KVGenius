"""
Command-line interface for the AI chatbot.
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
import logging

logger = logging.getLogger(__name__)


def main():
    """Main CLI application."""
    print("=" * 60)
    print("AI Chatbot - Powered by Hugging Face")
    print("=" * 60)
    
    # Load environment and configuration
    load_environment()
    config = load_config()
    setup_logging(config.get('app', {}).get('log_level', 'INFO'))
    
    print("\nInitializing chatbot...")
    
    try:
        # Load model
        model_config = config.get('model', {})
        model_loader = ModelLoader(
            model_name=model_config.get('name', 'microsoft/DialoGPT-medium'),
            cache_dir=model_config.get('cache_dir'),
            device=model_config.get('device', 'auto'),
            token=model_config.get('token')
        )
        
        model, tokenizer = model_loader.load()
        print(f"✓ Model loaded: {model_config.get('name')}")
        
        # Initialize chatbot
        chat_config = config.get('chat', {})
        chatbot = ChatBot(
            model=model,
            tokenizer=tokenizer,
            device=model_loader.device,
            max_history=chat_config.get('max_history', 5)
        )
        
        print("✓ Chatbot ready!")
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("Type 'reset' to clear conversation history.")
        print("Type 'help' for more commands.\n")
        
        # Privacy reminder
        print("🔒 Privacy: Model runs locally on your machine.")
        print(f"   Chat history saved to: {config.get('app', {}).get('history_file', 'chat_history.json')}\n")
        
        # Get generation parameters
        gen_config = config.get('generation', {})
        
        # Chat loop
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye! 👋")
                    break
                
                if user_input.lower() == 'reset':
                    chatbot.reset_conversation()
                    print("\n[Conversation history cleared]\n")
                    continue
                
                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  quit/exit - End the conversation")
                    print("  reset     - Clear conversation history")
                    print("  help      - Show this help message\n")
                    continue
                
                # Generate response
                response = chatbot.generate_response(
                    user_input,
                    max_length=gen_config.get('max_length', 100),
                    temperature=gen_config.get('temperature', 0.7),
                    top_k=gen_config.get('top_k', 50),
                    top_p=gen_config.get('top_p', 0.9),
                    repetition_penalty=gen_config.get('repetition_penalty', 1.2)
                )
                
                print(f"Bot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\nError: {e}\n")
    
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        print(f"\nError: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

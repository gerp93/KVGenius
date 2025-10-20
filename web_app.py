"""
Web interface for the AI chatbot using Gradio.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ModelLoader
from src.chat import ChatBot
from src.utils import load_config, setup_logging, load_environment
import gradio as gr
import logging

logger = logging.getLogger(__name__)

# Global variables for model and chatbot
chatbot_instance = None
gen_config = {}


def initialize_chatbot():
    """Initialize the chatbot model."""
    global chatbot_instance, gen_config
    
    load_environment()
    config = load_config()
    setup_logging(config.get('app', {}).get('log_level', 'INFO'))
    
    logger.info("Initializing chatbot...")
    
    # Load model
    model_config = config.get('model', {})
    model_loader = ModelLoader(
        model_name=model_config.get('name', 'microsoft/DialoGPT-medium'),
        cache_dir=model_config.get('cache_dir'),
        device=model_config.get('device', 'auto'),
        token=model_config.get('token')
    )
    
    model, tokenizer = model_loader.load()
    
    # Initialize chatbot
    chat_config = config.get('chat', {})
    chatbot_instance = ChatBot(
        model=model,
        tokenizer=tokenizer,
        device=model_loader.device,
        max_history=chat_config.get('max_history', 5)
    )
    
    gen_config = config.get('generation', {})
    logger.info("Chatbot initialized successfully")


def chat_fn(message, history):
    """
    Handle chat interaction.
    
    Args:
        message: User's message
        history: Chat history from Gradio
        
    Returns:
        Response message
    """
    if chatbot_instance is None:
        return "Error: Chatbot not initialized"
    
    try:
        response = chatbot_instance.generate_response(
            message,
            max_length=gen_config.get('max_length', 100),
            temperature=gen_config.get('temperature', 0.7),
            top_k=gen_config.get('top_k', 50),
            top_p=gen_config.get('top_p', 0.9),
            repetition_penalty=gen_config.get('repetition_penalty', 1.2)
        )
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


def reset_conversation():
    """Reset the conversation history."""
    if chatbot_instance:
        chatbot_instance.reset_conversation()
    return []


def main():
    """Main web application."""
    print("Initializing AI Chatbot Web Interface...")
    
    try:
        # Initialize chatbot
        initialize_chatbot()
        
        # Create Gradio interface
        with gr.Blocks(title="AI Chatbot") as demo:
            gr.Markdown("# 🤖 AI Chatbot")
            gr.Markdown("Powered by Hugging Face Transformers")
            
            chatbot_ui = gr.Chatbot(
                label="Chat",
                height=500,
                type="messages"
            )
            
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear History")
            
            # Handle message submission
            msg.submit(chat_fn, inputs=[msg, chatbot_ui], outputs=[chatbot_ui])
            submit_btn.click(chat_fn, inputs=[msg, chatbot_ui], outputs=[chatbot_ui])
            
            # Handle clear button
            clear_btn.click(reset_conversation, outputs=[chatbot_ui])
            
            gr.Markdown("---")
            gr.Markdown("💡 **Tip:** Type your message and press Enter or click Send")
        
        # Launch the app
        print("\n✓ Web interface ready!")
        print("Opening in browser...\n")
        demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
        
    except Exception as e:
        logger.error(f"Error starting web app: {e}")
        print(f"\nError: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

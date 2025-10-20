# AI Chatbot with Hugging Face 🤖

A Python-based AI chatbot application using Hugging Face transformers models for conversational AI. This project provides both CLI and web interfaces for interacting with various language models.

## Features

- 🚀 Support for multiple Hugging Face models (GPT-2, DialoGPT, BLOOM, etc.)
- 💬 Conversation history management
- 🖥️ Command-line interface (CLI)
- 🌐 Web interface using Gradio
- ⚙️ Configurable generation parameters
- 📝 Conversation history saving
- 🔧 Easy model switching via configuration

## Project Structure

```
KVGenius/
├── src/
│   ├── models/          # Model loading and management
│   │   ├── __init__.py
│   │   └── model_loader.py
│   ├── chat/            # Chat interface logic
│   │   ├── __init__.py
│   │   └── chatbot.py
│   └── utils/           # Utility functions
│       ├── __init__.py
│       └── config_helper.py
├── config/
│   └── config.yaml      # Configuration file
├── examples/
│   ├── simple_chat.py   # Simple usage example
│   └── test_models.py   # Model testing script
├── cli_app.py           # Command-line interface
├── web_app.py           # Web interface
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or navigate to the repository:**
   ```powershell
   cd c:\Users\kgerp\source\repos\KVGenius
   ```

2. **Create a virtual environment (recommended):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure environment (optional):**
   ```powershell
   copy .env.example .env
   # Edit .env file with your settings
   ```

## Usage

### Command-Line Interface (CLI)

Run the CLI application:
```powershell
python cli_app.py
```

Available commands:
- Type your message to chat
- `reset` - Clear conversation history
- `help` - Show available commands
- `quit` or `exit` - End the conversation

### Web Interface

Launch the web interface:
```powershell
python web_app.py
```

The interface will open in your browser at `http://127.0.0.1:7860`

### Examples

Run the simple example:
```powershell
python examples\simple_chat.py
```

Test different models:
```powershell
python examples\test_models.py
```

## Configuration

Edit `config/config.yaml` to customize:

### Model Settings
```yaml
model:
  name: "microsoft/DialoGPT-medium"  # Change model here
  cache_dir: "./model_cache"
  device: "auto"  # auto, cpu, or cuda
```

### Generation Parameters
```yaml
generation:
  max_length: 1000
  temperature: 0.7      # Higher = more random
  top_k: 50
  top_p: 0.9
  repetition_penalty: 1.2
```

### Chat Settings
```yaml
chat:
  max_history: 5  # Number of conversation turns to remember
  system_prompt: "You are a helpful AI assistant."
```

## Supported Models

Some popular models you can use:

- **DialoGPT** (Conversational):
  - `microsoft/DialoGPT-small`
  - `microsoft/DialoGPT-medium`
  - `microsoft/DialoGPT-large`

- **GPT-2** (General text):
  - `gpt2`
  - `gpt2-medium`
  - `gpt2-large`

- **BLOOM** (Multilingual):
  - `bigscience/bloom-560m`
  - `bigscience/bloom-1b1`

- **BlenderBot** (Conversational):
  - `facebook/blenderbot-400M-distill`

To use a different model, update the `model.name` in `config/config.yaml`.

## Environment Variables

Create a `.env` file from `.env.example`:

```env
# Hugging Face API Token (optional, for private models)
HUGGINGFACE_TOKEN=your_token_here

# Model Configuration
DEFAULT_MODEL=microsoft/DialoGPT-medium
MAX_LENGTH=1000
TEMPERATURE=0.7

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
```

## Development

### Running Tests

```powershell
pytest tests/
```

### Code Formatting

```powershell
black src/
```

### Linting

```powershell
flake8 src/
```

## Troubleshooting

### CUDA/GPU Issues
If you encounter GPU-related errors:
- Set `device: "cpu"` in `config/config.yaml`
- Or ensure PyTorch with CUDA support is installed

### Model Download Issues
- Models are cached in `./model_cache` by default
- First run may take time to download models
- Ensure stable internet connection

### Memory Issues
- Use smaller models (e.g., `DialoGPT-small` or `gpt2`)
- Reduce `max_length` in configuration
- Use CPU instead of GPU for smaller memory footprint

## Requirements

Key dependencies:
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for model inference
- `gradio` - Web interface framework
- `pyyaml` - Configuration file parsing
- `python-dotenv` - Environment variable management

See `requirements.txt` for complete list.

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Acknowledgments

- Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
- Web interface powered by [Gradio](https://gradio.app/)
- Uses pre-trained models from [Hugging Face Model Hub](https://huggingface.co/models)

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review configuration settings
3. Consult Hugging Face documentation
4. Open an issue in the repository

---

**Happy Chatting! 🎉**

"""
Model loader and manager for Hugging Face transformers models.
"""
from typing import Optional, Tuple
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages Hugging Face models for chatbot."""
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None, device: str = "auto", token: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            model_name: Name of the model from Hugging Face Hub
            cache_dir: Directory to cache downloaded models
            device: Device to load model on ('auto', 'cpu', or 'cuda')
            token: Hugging Face token (optional, for private/gated models)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = self._get_device(device)
        self.token = token or os.environ.get('HUGGINGFACE_TOKEN')
        self.model = None
        self.tokenizer = None
        
    def _get_device(self, device: str) -> str:
        """Determine the device to use for model."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            if self.token:
                logger.info("Using Hugging Face authentication token")
            
            # Warn if model name suggests a remote API
            if any(x in self.model_name.lower() for x in ['api', 'inference', 'endpoint', 'http://', 'https://']):
                logger.warning("⚠️  WARNING: Model name suggests remote API usage!")
                logger.warning("⚠️  Your chat data may be sent to remote servers.")
                logger.warning(f"⚠️  Model: {self.model_name}")
                print("\n" + "=" * 60)
                print("⚠️  PRIVACY WARNING")
                print("=" * 60)
                print(f"Model name suggests remote API: {self.model_name}")
                print("This may send your chat data to external servers.")
                print("Press Ctrl+C to cancel or wait 5 seconds to continue...")
                print("=" * 60 + "\n")
                import time
                time.sleep(5)
            else:
                logger.info("✓ Local model - chat data stays on your machine")
            
            # Load tokenizer. If the model repo contains custom code (new model classes),
            # `trust_remote_code=True` lets Transformers load that code. We pass the
            # authentication token if provided.
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                padding_side='left',
                token=self.token,
                trust_remote_code=True,
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try to load as causal LM first (for GPT-style models)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="cuda:0" if self.device == "cuda" else None,
                    token=self.token,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                logger.info("Loaded as CausalLM model")
            except Exception as e:
                # Fall back to seq2seq models (for T5, BART, etc.)
                logger.info(f"CausalLM loading failed ({type(e).__name__}), trying Seq2SeqLM")
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="cuda:0" if self.device == "cuda" else None,
                        token=self.token,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    logger.info("Loaded as Seq2SeqLM model")
                except Exception as e2:
                    # If both fail, raise the original error
                    raise e
            
            # No need to call .to() when using device_map
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_pipeline(self, task: str = "text-generation", **kwargs):
        """
        Create a Hugging Face pipeline for easier inference.
        
        Args:
            task: The task type (e.g., 'text-generation', 'conversational')
            **kwargs: Additional arguments for pipeline
            
        Returns:
            Hugging Face pipeline object
        """
        if self.model is None or self.tokenizer is None:
            self.load()
            
        return pipeline(
            task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            **kwargs
        )

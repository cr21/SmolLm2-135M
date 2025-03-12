import os
import yaml
import math
import logging
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW

from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ankita_model import create_smollm2_135m, LlamaConfig

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_tokenizer(config):
    """Setup tokenizer based on configuration"""
    tokenizer_config = config['tokenizer']
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config['tokenizer_name_or_path'],
        revision=tokenizer_config['tokenizer_revision'],
        model_max_length=config['tokens']['sequence_length'],
    )
    return tokenizer


def setup_model(config, device):
    """Setup model based on configuration"""
    model_config = config['model']['model_config']
    llama_config = LlamaConfig(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        intermediate_size=model_config['intermediate_size'],
        num_hidden_layers=model_config['num_hidden_layers'],
        num_attention_heads=model_config['num_attention_heads'],
        num_key_value_heads=model_config['num_key_value_heads'],
        hidden_act=model_config['hidden_act'],
        max_position_embeddings=model_config['max_position_embeddings'],
        initializer_range=model_config['initializer_range'],
        rms_norm_eps=model_config['rms_norm_eps'],
        use_cache=model_config['use_cache'],
        pad_token_id=model_config['pad_token_id'],
        bos_token_id=model_config['bos_token_id'],
        eos_token_id=model_config['eos_token_id'],
        tie_word_embeddings=model_config['tie_word_embeddings'],
        rope_theta=model_config['rope_theta'],
        rope_scaling=model_config['rope_scaling'],
        rope_interleaved=model_config['rope_interleaved'],
        pretraining_tp=model_config['pretraining_tp'],
        is_llama_config=model_config['is_llama_config'],
        use_return_dict=False,
    )
    
    model = create_smollm2_135m()
    
    # Apply initialization
    std = config['model']['init_method']['std']
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
    
    # Enable gradient checkpointing manually
    if config['parallelism'].get('recompute_layer', False):
        # Enable gradient checkpointing for model layers
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for layer in model.model.layers:
            layer.forward = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer.forward),
                use_reentrant=False
            )
    
    model = model.to(device)
    
    # For M2, use mps dtype
    if config['model']['dtype'] == 'bfloat16' and device.type == 'mps':
        # Note: as of the time of writing, bfloat16 might not be fully supported on MPS
        # Fallback to float16 or float32 based on compatibility
        if torch.backends.mps.is_available():
            try:
                model = model.to(torch.float16)  # Use float16 as fallback
                logger.info("Using float16 precision (bfloat16 requested but using float16 for MPS compatibility)")
            except Exception as e:
                logger.warning(f"Failed to use float16, falling back to float32: {e}")
                model = model.to(torch.float32)
                logger.info("Using float32 precision")
        else:
            model = model.to(torch.float32)
            logger.info("Using float32 precision (MPS not fully available)")
    
    return model


def setup_optimizer(config, model):
    """Setup optimizer and learning rate scheduler based on configuration"""
    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config['optimizer_factory']['name']
    lr = optimizer_config['learning_rate_scheduler']['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    adam_beta1 = optimizer_config['optimizer_factory']['adam_beta1']
    adam_beta2 = optimizer_config['optimizer_factory']['adam_beta2']
    adam_eps = optimizer_config['optimizer_factory']['adam_eps']
    
    # Filter parameters that should have weight decay applied
    decay_parameters = []
    no_decay_parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)
    
    optimizer_grouped_parameters = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0}
    ]
    
    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def get_lr_scheduler(config, optimizer, num_training_steps):
    """Get learning rate scheduler"""
    lr_config = config['optimizer']['learning_rate_scheduler']
    warmup_steps = lr_config['lr_warmup_steps']
    warmup_style = lr_config['lr_warmup_style']
    decay_style = lr_config['lr_decay_style']
    
    if decay_style == 'linear':
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        # Implement other schedulers as needed
        raise ValueError(f"Unsupported decay style: {decay_style}")


class FineWebDataset(IterableDataset):
    """Dataset class for the FineWeb dataset."""
    
    def __init__(self, config, tokenizer, split="train"):
        self.config = config
        self.tokenizer = tokenizer
        self.sequence_length = config['tokens']['sequence_length']
        self.split = split
        
        # Load the dataset
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb", 
            "CC-MAIN-2013-20",
            streaming=True
        )[split]
        
        # Filter for English text only
        self.dataset = self.dataset.filter(lambda x: x['language'] == 'en')
        
    def __iter__(self):
        """
        Iterator for FineWebDataset that properly handles sequence length and batching.
        Returns batches of tokenized text with attention masks and labels for training.
        """
        buffer = []
        buffer_size = 0
        current_example = []
        max_length = self.sequence_length

        for example in self.dataset:
            if not example.get('text'):
                continue

            text = example['text']

            # Skip very short texts (likely not useful for training)
            if len(text) < 100:
                continue

            # Tokenize the text without adding special tokens initially
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Add tokens to the current example
            current_example.extend(tokens)

            # Process complete sequences while we have enough tokens
            while len(current_example) >= max_length:
                # Take exactly sequence_length tokens
                sequence = current_example[:max_length]
                buffer.append(sequence)
                buffer_size += 1

                # Keep the remainder for next iteration
                current_example = current_example[max_length:]

                # Yield a batch when we have accumulated enough examples
                if buffer_size >= self.config['tokens']['micro_batch_size']:
                    # Convert to tensors
                    input_ids = torch.tensor(buffer, dtype=torch.long)

                    # Create labels (same as input_ids for causal LM)
                    labels = input_ids.clone()

                    # Create attention mask (all 1s since we have fixed length sequences)
                    attention_mask = torch.ones_like(input_ids)

                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }

                    # Reset buffer
                    buffer = []
                    buffer_size = 0

        # Handle any remaining examples in the buffer if we have at least one
        if buffer:
            # We might have a partial batch at the end
            input_ids = torch.tensor(buffer, dtype=torch.long)
            labels = input_ids.clone()
            attention_mask = torch.ones_like(input_ids)

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        # Handle any remaining tokens in current_example if long enough
        if len(current_example) >= max_length // 2:  # Only use if at least half the max length
            # Pad to max_length
            padding_length = max_length - len(current_example)
            padded_sequence = current_example + [self.tokenizer.pad_token_id] * padding_length

            input_ids = torch.tensor([padded_sequence], dtype=torch.long)

            # Create labels (same as input but with -100 for padding tokens so they're ignored in loss)
            labels = input_ids.clone()
            if padding_length > 0:
                labels[0, -padding_length:] = -100  # Set padding token labels to -100

            # Create attention mask (0 for padding tokens)
            attention_mask = torch.ones_like(input_ids)
            if padding_length > 0:
                attention_mask[0, -padding_length:] = 0

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }


def train(config, model, tokenizer, device):
    """Main training function"""
    
    # Create dataset and dataloader
    train_dataset = FineWebDataset(config, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,  # We handle batching in the dataset
        num_workers=config['data_stages'][0]['data'].get('num_loading_workers', 0),
    )
    
    # Setup optimizer and lr scheduler
    optimizer = setup_optimizer(config, model)
    
    # Calculate total training steps
    total_steps = config['tokens']['train_steps']
    
    scheduler = get_lr_scheduler(config, optimizer, total_steps)
    
    # Setup gradient accumulation
    gradient_accumulation_steps = config['tokens']['batch_accumulation_per_replica']
    
    # Setup mixed precision training - not using as MPS doesn't fully support autocast yet
    use_mixed_precision = (config['model']['dtype'] in ['bfloat16', 'float16']) and device.type == 'mps'
    
    # Create output directories
    os.makedirs(config['checkpoints']['checkpoints_path'], exist_ok=True)
    
    # Setup checkpoint saving
    checkpoint_interval = config['checkpoints']['checkpoint_interval']
    
    # Log training parameters
    logger.info(f"Starting training with the following parameters:")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Batch size: {config['tokens']['micro_batch_size']}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Sequence length: {config['tokens']['sequence_length']}")
    logger.info(f"Learning rate: {config['optimizer']['learning_rate_scheduler']['learning_rate']}")
    logger.info(f"Using mixed precision: {use_mixed_precision}")
    logger.info(f"Device: {device}")
    
    # Set model to training mode
    model.train()
    
    # Initialize running metrics
    running_loss = 0.0
    global_step = 0
    
    # Main training loop
    logger.info("Starting training...")
    progress_bar = tqdm(total=total_steps, desc="Training")
    
    # MPS specific info
    if device.type == 'mps':
        logger.info("Training on Apple Silicon GPU with Metal Performance Shaders (MPS)")
        
    for batch_idx, batch in enumerate(train_dataloader):
        # Stop if we reach the total number of steps
        if global_step >= total_steps:
            break
        
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass with gradient accumulation
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs[0]
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        running_loss += loss.item() * gradient_accumulation_steps
        
        # Only update weights after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            if config['optimizer']['clip_grad'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['optimizer']['clip_grad']
                )
            
            # Update weights
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Update global step
            global_step += 1
            
            # Log metrics
            if global_step % config['logging']['iteration_step_info_interval'] == 0:
                lr = scheduler.get_last_lr()[0]
                avg_loss = running_loss / config['logging']['iteration_step_info_interval']
                logger.info(f"Step {global_step}: loss={avg_loss:.4f}, lr={lr:.8f}")
                running_loss = 0.0
            
            # Save checkpoint
            if global_step % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    config['checkpoints']['checkpoints_path'], 
                    f"checkpoint-{global_step}"
                )
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Save model
                torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
                
                # Save optimizer and scheduler
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))
                
                logger.info(f"Saved checkpoint at step {global_step}")
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.8f}"})
    
    # Save final model
    if config['checkpoints']['save_final_state']:
        final_checkpoint_path = os.path.join(
            config['checkpoints']['checkpoints_path'], 
            "final_checkpoint"
        )
        os.makedirs(final_checkpoint_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(final_checkpoint_path, "model.pt"))
        logger.info(f"Saved final model")
    
    logger.info("Training completed!")


def verify_rotary_embeddings(model):
    """Verify that all layers have rotary embeddings set correctly"""
    for i, layer in enumerate(model.model.layers):
        if layer.self_attn._rotary_emb is None:
            raise ValueError(f"Layer {i}: Rotary embeddings not set!")
    print("All rotary embeddings correctly set!")


def main():
    parser = argparse.ArgumentParser(description="Train SmolLM2-135M model on Mac GPU")
    parser.add_argument("--config", type=str, default="config_smollm2_135M.yaml", help="Path to the YAML config file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU with Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("MPS not available, falling back to CPU")
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(config)
    
    # Setup model
    model = setup_model(config, device)
    
    # Verify rotary embeddings are set correctly
    verify_rotary_embeddings(model)
    
    # Set random seed for reproducibility
    torch.manual_seed(config['general']['seed'])
    
    # Train the model
    train(config, model, tokenizer, device)


if __name__ == "__main__":
    main()
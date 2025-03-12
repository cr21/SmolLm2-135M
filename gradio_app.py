import gradio as gr
import torch
from transformers import AutoTokenizer
import yaml
from SmolLm3 import LlamaModel


def generate_helper(model, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, eos_token=None, device=None):
    
    model = model.to(device)
    idx = idx.to(device)
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits, _ = model(idx_cond)  # Unpack both logits and loss (ignore loss)
            logits = logits.view(idx_cond.shape[0], -1, model.config['vocab_size'])  # Reshape to [batch, seq, vocab]
            
        # Get the logits for the last token only
        logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        
        if top_k is not None:
            # top k sampling
            top_logits, top_pos = torch.topk(logits, top_k)
            min_logit = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_logit,
                               torch.tensor(float('-inf')).to(logits.device),
                               logits)
        
        # temperature scaling
        if temperature > 0.0:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        if idx_next.item() == eos_token:
            break
            
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return idx

def get_config(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    return config

def load_model_from_checkpoint(config_path, checkpoint_path, device):
    config = get_config(config_path)
    model = LlamaModel(config['model'])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def load_weights(config, weights_path, device):
    model = LlamaModel(config['model'])
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    return model

def get_tokenizer(config):
    tokenizer_path = config['tokenizer']['tokenizer_name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    return tokenizer, vocab_size

def generate_text(model, tokenizer, input_text, max_new_tokens, context_length, temperature, top_k, eos_token, device):
    encoded_text = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generated_text = generate_helper(model, 
                            idx=encoded_text,
                            max_new_tokens=max_new_tokens,
                            context_length=context_length, 
                            temperature=temperature, 
                            top_k=top_k, 
                            eos_token=eos_token, 
                            device=device)
    return tokenizer.decode(generated_text.squeeze(0))



# Initialize model and tokenizer
def initialize_model():
    config_path = "config_smollm2_135M.yaml"
    checkpoint_path = "/Users/chiragtagadiya/Documents/Final_training_before_stop_smolllm3/checkpoints/model_37000_steps_avg_loss_2.85920_optimizer_lr_0.00000003.pth"  # Update this path
    weights_path = "model_weights_35000_step.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configuration
    config = get_config(config_path)
    
    # Load model
    # model = load_model_from_checkpoint(config_path, checkpoint_path, device)
    model = load_weights(config, weights_path, device)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer, vocab_size = get_tokenizer(config)
    
    return model, tokenizer, device

def generate_response(prompt, max_new_tokens):
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text=prompt,
        max_new_tokens=max_new_tokens,
        context_length=256,
        temperature=0.9,
        top_k=2,
        eos_token=tokenizer.eos_token_id,
        device=device
    )
    return generated_text

# Initialize global variables
model, tokenizer, device = initialize_model()

# Create Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(
            lines=3, 
            placeholder="Enter your prompt here...",
            label="Input Prompt"
        ),
        gr.Slider(
            minimum=50,
            maximum=256,
            value=100,
            step=10,
            label="Max New Tokens"
        )
    ],
    outputs=gr.Textbox(
        lines=5,
        label="Generated Text"
    ),
    title="SmolLM Text Generator",
    description="Enter a prompt and adjust the maximum number of tokens to generate text with SmolLM model."
)

if __name__ == "__main__":
    iface.launch()

import torch
from gpt_like import encode, decode, GPTLanguageModel, vocab_size
from transformers import  PreTrainedModel, PretrainedConfig

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


model = GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load('gpt1', weights_only=True))
model.eval()


class GPTConfig(PretrainedConfig):
    model_type = "gpt_custom"

class GPTForHF(PreTrainedModel):
    config_class = GPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = GPTLanguageModel(vocab_size)  # Your model inside Hugging Face format

    def forward(self, input_ids):
        return self.model(input_ids)

# Convert PyTorch model to Hugging Face
config = GPTConfig()
hf_model = GPTForHF(config)
hf_model.model.load_state_dict(torch.load('gpt1', weights_only=True))
hf_model.eval()
hf_model.save_pretrained("hf_model")

# This is a valid hugging face model, but unfortunately is not convertible to GGUF with llama.cpp because its architecture is not recognized. 
# So we cannot open it with LMStudio


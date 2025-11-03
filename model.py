"""
Transformer-based Binary Classifier with LoRA support
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

class TransformerBinaryClassifier(nn.Module):
    """
    Transformer-based binary classifier with optional LoRA fine-tuning.
    """
    def __init__(self, model_name, dropout=0.1, use_lora=True, 
                 lora_r=8, lora_alpha=16, lora_dropout=0.05):
        """
        Initialize the binary classifier.
        
        Args:
            model_name (str): Name or path of pre-trained transformer model
            dropout (float): Dropout rate for classification head
            use_lora (bool): Whether to apply LoRA to the model
            lora_r (int): LoRA rank (smaller = fewer parameters)
            lora_alpha (int): LoRA scaling factor
            lora_dropout (float): LoRA dropout rate
        """
        super(TransformerBinaryClassifier, self).__init__()
        
        # Load base encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,  # Sequence classification task
                r=lora_r,  # Rank of LoRA matrices
                lora_alpha=lora_alpha,  # Scaling factor
                lora_dropout=lora_dropout,  # Dropout for LoRA layers
                bias="none",  # Can be "none", "all", or "lora_only"
                target_modules=["query", "value"],  # Apply LoRA to attention layers
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()  # See trainable params
        
        # Classification head (remains fully trainable)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through encoder and classifier."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            labels = labels.view(-1, 1)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}
    
    def freeze_base_layers(self):
        """Freeze base encoder (not needed with LoRA, but kept for compatibility)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_base_layers(self):
        """Unfreeze base encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True

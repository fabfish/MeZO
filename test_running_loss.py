#!/usr/bin/env python3
"""
Test script to demonstrate the running loss logging functionality.
This script shows how the training progress will now display running loss and learning rate.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments
from large_models.trainer import OurTrainer

# Create a simple dummy dataset for testing
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 1000, (10,)),
            'labels': torch.randint(0, 100, (1,)).squeeze()
        }

# Create a simple dummy model for testing
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 128)
        self.linear = nn.Linear(128, 100)
    
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = x.mean(dim=1)  # Average over sequence length
        logits = self.linear(x)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}

def test_running_loss_logging():
    """Test the running loss logging functionality."""
    print("Testing running loss logging functionality...")
    
    # Create dummy data
    dataset = DummyDataset(size=50)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create model
    model = DummyModel()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=1e-3,
        logging_steps=10,
        disable_tqdm=False,  # Enable tqdm to see the progress bar
        report_to=[],  # Disable wandb/tensorboard for testing
        save_strategy="no",
        evaluation_strategy="no",
    )
    
    # Create trainer
    trainer = OurTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("\nStarting training with running loss logging...")
    print("You should see a progress bar that shows:")
    print("- Epoch progress (e.g., Epoch 1/2)")
    print("- Running loss (average loss per step)")
    print("- Learning rate")
    print("- Progress percentage and time estimates")
    print("\n" + "="*60 + "\n")
    
    # Start training - this will show the running loss in the progress bar
    trainer.train()
    
    print("\n" + "="*60 + "\n")
    print("Training completed!")
    print("The running loss logging is now active and will show:")
    print("- Real-time loss values during training")
    print("- Current learning rate")
    print("- Progress through each epoch")

if __name__ == "__main__":
    test_running_loss_logging()

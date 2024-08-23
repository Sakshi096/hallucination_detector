
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from dataset import HallucinationDataset
from model import HallucinationDetector
from data import data
from transformers import BertTokenizer

# Prepare data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = HallucinationDataset(data, tokenizer, max_len=128)
train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# Initialize model, optimizer, and loss function
model = HallucinationDetector()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}")

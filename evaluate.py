
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import HallucinationDetector
from dataset import HallucinationDataset
from data import data
from transformers import BertTokenizer

# Prepare data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = HallucinationDataset(data, tokenizer, max_len=128)
val_loader = DataLoader(dataset, batch_size=8, shuffle=False)

# Load model
model = HallucinationDetector()
model.eval()

# Evaluation
predictions, true_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)

        predictions.extend(preds)
        true_labels.extend(labels)

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

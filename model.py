# model.py

from transformers import BertForSequenceClassification

class HallucinationDetector(nn.Module):
    def __init__(self, n_classes=2):
        super(HallucinationDetector, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

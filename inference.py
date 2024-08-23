
from model import HallucinationDetector
from transformers import BertTokenizer

def detect_hallucination(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = HallucinationDetector()
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
    
    return "hallucinated" if preds.item() == 1 else "correct"

# Example usage
if __name__ == "__main__":
    test_text = "The capital of the United States is New York."
    result = detect_hallucination(test_text)
    print(f"The statement is: {result}")

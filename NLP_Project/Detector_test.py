import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack"
]

def predict(model, tokenizer, texts, device):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
    return probs


def compare_models(model_before, model_after, tokenizer, texts, device):
    probs_before = predict(model_before, tokenizer, texts, device)
    probs_after = predict(model_after, tokenizer, texts, device)

    for i, text in enumerate(texts):
        print("\n" + "=" * 60)
        print(f"Input: {text}")
        print("=" * 60)
        print(f"{'Label':20s} | {'Before':>10} | {'After':>10} | {'Change':>10}")
        print("-" * 60)
        for j, label in enumerate(LABELS):
            before = probs_before[i][j]
            after = probs_after[i][j]
            change = after - before
            direction = "1 +" if change > 0.01 else ("2 " if change < -0.01 else "3  ")
            print(f"{label:20s} | {before:>10.4f} | {after:>10.4f} | {direction}{change:+.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load both models — both now trained on Kaggle data
    tokenizer = BertTokenizer.from_pretrained("Saved_model_baseline")
    model_before = BertForSequenceClassification.from_pretrained("Saved_model_baseline").to(device)
    model_after = BertForSequenceClassification.from_pretrained("Saved_model_fair").to(device)

    texts = [
        # Batch of Texts to Test
    ]

    print("\n COMPARING MODELS: BEFORE vs AFTER BIAS MITIGATION")
    print("1 = toxicity went DOWN after mitigation")
    print("2 = toxicity went UP after mitigation")
    print("3  = no significant change")

    compare_models(model_before, model_after, tokenizer, texts, device)


if __name__ == "__main__":
    main()

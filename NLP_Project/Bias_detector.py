import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

LABEL_COLS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack"
]

LABEL_THRESHOLD = 0.5


def tokenize_and_encode(tokenizer, texts, labels, max_length=128):
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    label_array = labels.values.astype(np.float32)
    label_tensor = torch.tensor((label_array >= LABEL_THRESHOLD).astype(np.float32))
    return input_ids, attention_masks, label_tensor


def compute_pos_weights(labels_df):
    binary = (labels_df.values >= LABEL_THRESHOLD).astype(np.float32)
    n_pos = binary.sum(axis=0)
    n_neg = len(binary) - n_pos
    pos_weight = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1), 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def train_model(model, train_loader, val_loader, optimizer, loss_fct, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fct(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fct(outputs.logits, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Val Loss:   {val_loss / len(val_loader):.4f}")
        print("-" * 40)


def evaluate_model(model, test_loader, device):
    model.eval()
    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)
            predicted_probs.append(probs.cpu())
            true_labels.append(labels.cpu())

    true_labels = torch.cat(true_labels, dim=0).numpy().astype(int)
    predicted_probs = torch.cat(predicted_probs, dim=0).numpy()

    thresholds = np.array([0.5, 0.3, 0.5, 0.3, 0.5, 0.3])
    predicted_labels = (predicted_probs > thresholds).astype(int)

    precision = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)

    print("\n== FINAL RESULTS (BASELINE - NO BIAS MITIGATION) ==")
    print(f"Precision (micro): {precision:.4f}")
    print(f"Recall (micro):    {recall:.4f}")
    print(f"F1 Score (micro):  {f1:.4f}")

    print("\n--- Per-Class F1 ---")
    f1_per_class = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
    for label, score in zip(LABEL_COLS, f1_per_class):
        print(f"  {label:20s}: {score:.4f}")


def main():
    warnings.filterwarnings("ignore")

    df = pd.read_csv("civil_comments_train.csv")

    df = df.rename(columns={"comment_text": "text", "target": "toxicity"})

    keep_cols = ["text"] + LABEL_COLS
    df = df[keep_cols].dropna()
    df = df.sample(n=50000, random_state=42)

    labels = df[LABEL_COLS]

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"], labels, test_size=0.25, random_state=42
    )
    test_texts, val_texts, test_labels, val_labels = train_test_split(
        test_texts, test_labels, test_size=0.5, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=6,
        problem_type="multi_label_classification"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)

    train_inputs = tokenize_and_encode(tokenizer, train_texts, train_labels)
    test_inputs = tokenize_and_encode(tokenizer, test_texts, test_labels)
    val_inputs = tokenize_and_encode(tokenizer, val_texts, val_labels)

    train_dataset = TensorDataset(*train_inputs)
    test_dataset = TensorDataset(*test_inputs)
    val_dataset = TensorDataset(*val_inputs)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

    pos_weights = compute_pos_weights(train_labels).to(device)
    loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    train_model(model, train_loader, val_loader, optimizer, loss_fct, device, num_epochs=3)

    model.save_pretrained("Saved_model_baseline")
    tokenizer.save_pretrained("Saved_model_baseline")

    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()

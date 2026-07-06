from datasets import load_dataset
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
    "target",           
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack"
]

DEMOGRAPHIC_COLS = [
    "female",
    "male",
    "black",
    "white",
    "muslim",
    "jewish",
    "christian",
    "latino"
]

LABEL_THRESHOLD = 0.5
DEMOGRAPHIC_THRESHOLD = 0.5
LAMBDA_BIAS = 0.5  


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


def demographic_parity_loss(logits, sensitive_attrs):

    probs = torch.sigmoid(logits)  
    total_loss = torch.tensor(0.0, device=logits.device)
    count = 0

    for i in range(sensitive_attrs.shape[1]):
        group_mask = sensitive_attrs[:, i] 

        group_1 = probs[group_mask == 1]   
        group_0 = probs[group_mask == 0] 

        if len(group_1) == 0 or len(group_0) == 0:
            continue

        mean_1 = group_1.mean(dim=0)
        mean_0 = group_0.mean(dim=0)

        total_loss += torch.norm(mean_1 - mean_0, p=2)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=logits.device)

    return total_loss / count


def train_model(model, train_loader, val_loader, optimizer, loss_fct, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_bias_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels, sensitive_attrs = [t.to(device) for t in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)

            cls_loss = loss_fct(outputs.logits, labels)

            bias_loss = demographic_parity_loss(outputs.logits, sensitive_attrs)

            loss = cls_loss + LAMBDA_BIAS * bias_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_bias_loss += bias_loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels, sensitive_attrs = [t.to(device) for t in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                cls_loss = loss_fct(outputs.logits, labels)
                bias_loss = demographic_parity_loss(outputs.logits, sensitive_attrs)
                loss = cls_loss + LAMBDA_BIAS * bias_loss
                val_loss += loss.item()

        print(f"Epoch {epoch+1}")
        print(f"Train Loss:     {total_loss / len(train_loader):.4f}")
        print(f"  Cls Loss:     {total_cls_loss / len(train_loader):.4f}")
        print(f"  Bias Loss:    {total_bias_loss / len(train_loader):.4f}")
        print(f"Val Loss:       {val_loss / len(val_loader):.4f}")
        print("-" * 40)


def evaluate_model(model, test_loader, device):
    model.eval()
    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels, sensitive_attrs = [t.to(device) for t in batch]
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

    print("\n==FINAL RESULTS (WITH BIAS MITIGATION)==")
    print(f"Precision (micro): {precision:.4f}")
    print(f"Recall (micro):    {recall:.4f}")
    print(f"F1 Score (micro):  {f1:.4f}")

    print("\n--- Per-Class F1 ---")
    f1_per_class = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
    for label, score in zip(LABEL_COLS, f1_per_class):
        print(f"  {label:20s}: {score:.4f}")


def evaluate_fairness(model, test_loader, device):

    model.eval()
    all_probs = []
    all_sensitive = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels, sensitive_attrs = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)
            all_probs.append(probs.cpu())
            all_sensitive.append(sensitive_attrs.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_sensitive = torch.cat(all_sensitive, dim=0).numpy()

    print("\n== FAIRNESS METRICS ==")
    print(f"{'Group':20s} | {'Avg Toxicity Pred':>18} | {'Count':>6}")
    print("-" * 50)

    for i, group in enumerate(DEMOGRAPHIC_COLS):
        mask = all_sensitive[:, i] == 1
        if mask.sum() == 0:
            continue
        avg_pred = all_probs[mask, 0].mean()  # toxicity column
        print(f"{group:20s} | {avg_pred:>18.4f} | {mask.sum():>6}")

    print("-" * 50)
    overall_avg = all_probs[:, 0].mean()
    print(f"{'Overall':20s} | {overall_avg:>18.4f} | {len(all_probs):>6}")


def main():
    warnings.filterwarnings("ignore")

    df = pd.read_csv("train.csv")

    df = df.rename(columns={"comment_text": "text", "target": "toxicity"})

    LABEL_COLS_LOCAL = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack"
    ]

    keep_cols = ["text"] + LABEL_COLS_LOCAL + DEMOGRAPHIC_COLS
    df = df[keep_cols].dropna()
    df = df.sample(n=50000, random_state=42)

    labels = df[LABEL_COLS_LOCAL]
    demographics = df[DEMOGRAPHIC_COLS]

    train_texts, test_texts, train_labels, test_labels, train_demo, test_demo = train_test_split(
        df["text"], labels, demographics, test_size=0.25, random_state=42
    )
    test_texts, val_texts, test_labels, val_labels, test_demo, val_demo = train_test_split(
        test_texts, test_labels, test_demo, test_size=0.5, random_state=42
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

    train_demo_tensor = torch.tensor((train_demo.values >= DEMOGRAPHIC_THRESHOLD).astype(np.float32))
    test_demo_tensor = torch.tensor((test_demo.values >= DEMOGRAPHIC_THRESHOLD).astype(np.float32))
    val_demo_tensor = torch.tensor((val_demo.values >= DEMOGRAPHIC_THRESHOLD).astype(np.float32))

    train_dataset = TensorDataset(*train_inputs, train_demo_tensor)
    test_dataset = TensorDataset(*test_inputs, test_demo_tensor)
    val_dataset = TensorDataset(*val_inputs, val_demo_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

    pos_weights = compute_pos_weights(train_labels).to(device)
    loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    train_model(model, train_loader, val_loader, optimizer, loss_fct, device, num_epochs=3)

    model.save_pretrained("Saved_model_fair")
    tokenizer.save_pretrained("Saved_model_fair")

    evaluate_model(model, test_loader, device)
    evaluate_fairness(model, test_loader, device)


if __name__ == "__main__":
    main()

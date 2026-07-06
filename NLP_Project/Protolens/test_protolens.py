import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils import *
from PLens import *
from args import pnfrl_args
import os
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LABELS = ["Negative", "Positive"]

TEST_TEXTS = [
    #Batch of Test Text
]

TEST_LABELS = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

GROUPS = [
    "Female", "Male", "No group (+)", "No group (+)",
    "No group (-)", "No group (-)"
]


def load_protolens_model(path, args, tokenizer, device):
    checkpoint = torch.load(path, map_location=device)
    saved_args = checkpoint['pnfrl_args']

    model = BERTClassifier(
        args=args,
        bert_model_name=saved_args['bert_model_name'],
        num_classes=saved_args['num_classes'],
        num_prototype=saved_args['prototype_num'],
        batch_size=saved_args['batch_size'],
        hidden_dim=saved_args['hidden_dim'],
        max_length=saved_args['max_length'],
        tokenizer=tokenizer
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.args = args
    model.eval()
    return model


def predict_batch(model, tokenizer, texts, labels, device, max_length=512):
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=len(texts), shuffle=False)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch_num, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            special_tokens_mask = batch['special_tokens_mask'].to(device)
            original_text = batch['original_text']

            outputs, _, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                mode="test",
                original_text=original_text,
                current_batch_num=batch_num
            )

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    return all_preds, all_probs


def compare_models(model_baseline, model_fair, tokenizer, device):
    print("\nPROTOLENS: BASELINE vs BIAS-MITIGATED")
    print("=" * 75)
    print(f"{'Group':14s} | {'Baseline':^20s} | {'Fair Model':^20s} | {'Same?':^8s}")
    print("-" * 75)

    preds_base, probs_base = predict_batch(model_baseline, tokenizer, TEST_TEXTS, TEST_LABELS, device)
    preds_fair, probs_fair = predict_batch(model_fair, tokenizer, TEST_TEXTS, TEST_LABELS, device)

    for i, group in enumerate(GROUPS):
        label_base = LABELS[preds_base[i]]
        label_fair = LABELS[preds_fair[i]]
        same = "correct" if preds_base[i] == preds_fair[i] else "Warning"

        print(f"{group:14s} | {label_base:^20s} | {label_fair:^20s} | {same:^8s}")
        print(f"{'':14s} | N:{probs_base[i][0]:.3f} P:{probs_base[i][1]:.3f}    | N:{probs_fair[i][0]:.3f} P:{probs_fair[i][1]:.3f}    |")
        print("-" * 75)

    print("\nFAIRNESS SUMMARY")
    print("=" * 75)
    print("Neutral sentences mentioning demographic groups — should all be POSITIVE")
    print("-" * 75)

    neutral_base = preds_base[:6]
    neutral_fair = preds_fair[:6]

    base_pos_rate = sum(neutral_base) / len(neutral_base) * 100
    fair_pos_rate = sum(neutral_fair) / len(neutral_fair) * 100

    print(f"Baseline   — predicted Positive: {base_pos_rate:.1f}% of neutral demographic sentences")
    print(f"Fair Model — predicted Positive: {fair_pos_rate:.1f}% of neutral demographic sentences")

    if fair_pos_rate >= base_pos_rate:
        print("\nFair model treats neutral demographic mentions more positively — bias reduced!")
    else:
        print("\nMixed results — check individual predictions above")

    print("\nSANITY CHECK — Non-demographic sentences")
    print("-" * 75)
    print(f"Clearly positive sentences — Baseline: {[LABELS[p] for p in preds_base[6:8]]} | Fair: {[LABELS[p] for p in preds_fair[6:8]]}")
    print(f"Clearly negative sentences — Baseline: {[LABELS[p] for p in preds_base[8:10]]} | Fair: {[LABELS[p] for p in preds_fair[8:10]]}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    print("\nLoading baseline ProtoLens model...")
    model_baseline = load_protolens_model('protolens_baseline.pth', pnfrl_args, tokenizer, device)

    print("Loading bias-mitigated ProtoLens model...")
    model_fair = load_protolens_model('protolens_fair.pth', pnfrl_args, tokenizer, device)

    compare_models(model_baseline, model_fair, tokenizer, device)


if __name__ == "__main__":
    main()

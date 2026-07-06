import pandas as pd

df = pd.read_csv("civil_comments_train.csv")
df = df.rename(columns={"comment_text": "text", "target": "toxicity"})

LABEL_COLS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
DEMOGRAPHIC_COLS = ["female", "male", "black", "white", "muslim", "jewish", "christian", "latino"]

keep_cols = ["text"] + LABEL_COLS + DEMOGRAPHIC_COLS
df_clean = df[keep_cols].dropna()
print(f"Rows after dropna: {len(df_clean)}")

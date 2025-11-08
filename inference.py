import pandas as pd
from transformers import pipeline

# Load your labeled CSV
data_path = "/home/premananda/Documents/Data/reliable_polish_news_labeled.csv"
df = pd.read_csv(data_path)

# Load BERT classifier
classifier = pipeline(
    "text-classification",
    model="dkleczek/bert-base-polish-uncased-v1",
    tokenizer="dkleczek/bert-base-polish-uncased-v1"
)

# Run inference on all texts
results = []
for text in df["text"]:
    res = classifier(text[:512])[0]  # [:512] to avoid token limit overflow
    results.append(res)

# Add predictions to the dataframe
df["predicted_label"] = [r["label"] for r in results]
df["confidence"] = [r["score"] for r in results]

# Save with predictions
output_path = "/home/premananda/Documents/Data/reliable_polish_news_with_predictions.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"✅ Predictions saved to: {output_path}")
print(df.head())

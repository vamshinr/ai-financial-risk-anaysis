import os
import glob
import nltk
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from tqdm import tqdm

nltk.download("punkt", quiet=True)

EXTRACTED_DIR = "extracted_text"
MODEL_DIR = "./fine_tuned_bert"
OUTPUT_DIR = "output/risk_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Label map
id2label = {0: "negative", 1: "neutral", 2: "positive"}

def assess_risk(text_file):
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = nltk.sent_tokenize(text)
    risks = []
    for sentence in tqdm(sentences, desc=f"Classifying sentences in {os.path.basename(text_file)}"):
        if len(sentence) > 10:
            result = classifier(sentence, truncation=True, max_length=512)[0]
            label_id = int(result["label"].split("_")[-1])
            label = id2label[label_id]
            if label == "negative":
                risks.append(sentence)
    return risks

def generate_report():
    text_files = glob.glob(f"{EXTRACTED_DIR}/*.txt")
    for text_file in text_files:
        risks = assess_risk(text_file)
        output_file = os.path.join(OUTPUT_DIR, os.path.basename(text_file).replace(".txt", "_risk_report.txt"))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Identified Risk Patterns (Negative Sentiment Sentences):\n\n")
            for risk in risks:
                f.write(f"- {risk}\n")
        print(f"Risk report generated: {output_file}")

if __name__ == "__main__":
    generate_report()

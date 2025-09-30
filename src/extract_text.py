import os
import glob
from bs4 import BeautifulSoup
from tqdm import tqdm

FILINGS_DIR = "filings"
EXTRACTED_DIR = "extracted_text"
os.makedirs(EXTRACTED_DIR, exist_ok=True)

def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text(separator="\n").strip()
    return text

def process_filings():
    html_files = glob.glob(f"{FILINGS_DIR}/**/*.html", recursive=True)
    for html_file in tqdm(html_files, desc="Extracting text"):
        text = extract_text_from_html(html_file)
        output_file = os.path.join(EXTRACTED_DIR, os.path.basename(html_file).replace(".html", ".txt"))
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
    print(f"Extracted text to {EXTRACTED_DIR}")

if __name__ == "__main__":
    process_filings()

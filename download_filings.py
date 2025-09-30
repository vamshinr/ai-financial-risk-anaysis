import os
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

# Directory to save filings
FILINGS_DIR = "filings"
os.makedirs(FILINGS_DIR, exist_ok=True)

# Sample companies (tickers and CIKs)
companies = [
    {"ticker": "AAPL", "cik": "0000320193"},
    {"ticker": "MSFT", "cik": "0000789019"},
    {"ticker": "GOOGL", "cik": "0001652044"},
    {"ticker": "AMZN", "cik": "0001018724"},
    {"ticker": "TSLA", "cik": "0001318605"}
]

# SEC EDGAR base URL
BASE_URL = "https://www.sec.gov"

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}

def get_filing_urls(cik, form_type="10-K", limit=1):
    """Scrape EDGAR for filing URLs for a given CIK and form type."""
    # Search URL for filings
    search_url = f"{BASE_URL}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&count={limit}&output=atom"
    
    try:
        response = requests.get(search_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")
        
        # Find all entry links
        entries = soup.find_all("entry")
        filing_urls = []
        
        for entry in entries[:limit]:
            filing_url = entry.find("link", rel="alternate")["href"]
            # Convert to document page URL
            filing_urls.append(filing_url)
        
        return filing_urls
    except requests.RequestException as e:
        print(f"Error fetching filings for CIK {cik}: {e}")
        return []

def get_document_url(filing_url):
    """Scrape the filing page to find the actual 10-K document URL."""
    try:
        response = requests.get(filing_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find table with documents
        table = soup.find("table", class_="tableFile")
        if not table:
            return None
        
        # Look for the primary 10-K document (HTML or HTM)
        for row in table.find_all("tr")[1:]:  # Skip header row
            cols = row.find_all("td")
            if len(cols) >= 3 and "10-K" in cols[2].text:
                doc_link = cols[1].find("a")
                if doc_link and doc_link["href"].endswith((".html", ".htm")):
                    return BASE_URL + doc_link["href"]
        return None
    except requests.RequestException as e:
        print(f"Error accessing filing page {filing_url}: {e}")
        return None

def download_filing(doc_url, ticker, cik):
    """Download the filing HTML content and save to disk."""
    try:
        response = requests.get(doc_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        # Generate a filename (e.g., AAPL_10K_YYYYMMDD.html)
        filing_date = re.search(r"\d{8}", doc_url)
        filing_date = filing_date.group(0) if filing_date else "unknown"
        filename = f"{ticker}_10K_{filing_date}.html"
        output_path = os.path.join(FILINGS_DIR, filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Saved filing: {output_path}")
        return True
    except requests.RequestException as e:
        print(f"Error downloading {doc_url}: {e}")
        return False

def download_filings():
    """Main function to download filings for all companies."""
    for company in tqdm(companies, desc="Downloading 10-K filings"):
        ticker = company["ticker"]
        cik = company["cik"]
        
        # Get filing URLs
        filing_urls = get_filing_urls(cik, form_type="10-K", limit=1)
        
        for filing_url in filing_urls:
            # Get the actual document URL
            doc_url = get_document_url(filing_url)
            if doc_url:
                # Download the filing
                download_filing(doc_url, ticker, cik)
                # Respect SEC rate limits (~10 requests per second)
                time.sleep(0.1)
    
    print(f"Downloaded filings to {FILINGS_DIR}")

if __name__ == "__main__":
    download_filings()

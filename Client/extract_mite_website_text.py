import requests
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import io
from urllib.parse import urljoin, urlparse

# Setup OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

START_URL = "https://spyd.vercel.app/"   # ✅ replace with your college site
OUTPUT_FILE = "mite_website_full_content.txt"

visited = set()
all_text = []

def extract_text_from_url(url):
    """Scrape visible text + OCR from images on a given webpage"""
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove scripts and styles
        for script in soup(["script", "style", "noscript"]):
            script.extract()

        # Extract visible text
        page_text = soup.get_text(separator="\n", strip=True)
        if page_text.strip():
            all_text.append(f"\n--- Page: {url} ---\n{page_text}")

        # Extract text from images
        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
            img_url = urljoin(url, src)  # handle relative URLs
            try:
                img_resp = requests.get(img_url, timeout=5)
                image = Image.open(io.BytesIO(img_resp.content))
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    all_text.append(f"\n[Image OCR from {img_url}]\n{ocr_text}")
            except Exception as e:
                print(f"⚠️ Could not OCR image {img_url}: {e}")
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")

def crawl_site(start_url, depth=2):
    """Recursive crawler to visit links up to given depth"""
    if depth == 0 or start_url in visited:
        return
    visited.add(start_url)

    extract_text_from_url(start_url)

    try:
        r = requests.get(start_url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        domain = urlparse(start_url).netloc  # only crawl same domain

        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(start_url, href)

            if urlparse(full_url).netloc == domain and full_url not in visited:
                crawl_site(full_url, depth - 1)
    except Exception as e:
        print(f"⚠️ Could not crawl links from {start_url}: {e}")

if __name__ == "__main__":
    print("🌐 Crawling website deeply...")
    crawl_site(START_URL, depth=3)  # depth=3 → homepage + subpages + deeper

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))

    print(f"✅ Full website content saved to {OUTPUT_FILE}")

import requests
from bs4 import BeautifulSoup
import csv
import time

BASE_URL    = "https://www.shl.com"
CATALOG_URL = BASE_URL + "/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def get_catalog_page(page_num: int):
    """
    Fetch the HTML for a given catalog page number.
    """
    params = {"page": page_num}
    resp = requests.get(CATALOG_URL, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.text

def parse_table_on_page(html: str):
    """
    Parse one page’s table rows into a list of dicts (without Details).
    """
    soup = BeautifulSoup(html, "lxml")
    wrapper = soup.find("div", class_="custom__table-wrapper")
    if not wrapper:
        return []

    rows = wrapper.find_all("tr")[1:]  # skip header
    page_data = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        # Title + link
        a = cols[0].find("a")
        title = a.get_text(strip=True)
        link  = BASE_URL + a["href"]

        # remote & adaptive
        remote  = "Yes" if cols[1].find("span", class_="catalogue__circle -yes") else "No"
        adaptive= "Yes" if cols[2].find("span", class_="catalogue__circle -yes") else "No"

        # test types
        types = [span.get_text(strip=True)
                 for span in cols[3].find_all("span", class_="product-catalogue__key")]
        test_types = ", ".join(types)

        page_data.append({
            "title": title,
            "link": link,
            "remote_testing": remote,
            "adaptive_irt": adaptive,
            "test_type": test_types
        })

    return page_data

def get_solution_details(url: str):
    try:
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        content = (
            soup.find("div", class_="product-page-content")
            or soup.find("article")
            or soup.find("main")
        )
        if not content:
            return "Details not found"

        paras = content.find_all("p")
        return "\n".join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
    except:
        return "Error fetching details"

def scrape_all_pages():
    all_items = []
    page = 1

    while True:
        print(f"→ Fetching page {page}…")
        html = get_catalog_page(page)
        items = parse_table_on_page(html)
        if not items:
            print("No more rows found; stopping pagination.")
            break

        print(f"   Found {len(items)} items on page {page}")
        all_items.extend(items)
        page += 1
        time.sleep(0.5)  # be polite

    return all_items

def main():
    # 1) Scrape table across all pages
    catalog = scrape_all_pages()
    print(f"\nTotal assessments found: {len(catalog)}\n")

    # 2) Fetch details and print row‐wise
    for idx, item in enumerate(catalog, start=1):
        details = get_solution_details(item["link"])
        print(f"\n[{idx}] {item['title']}")
        print(f"Remote Testing: {item['remote_testing']}")
        print(f"Adaptive/IRT:   {item['adaptive_irt']}")
        print(f"Test Type:      {item['test_type']}")
        print(f"Details:\n{details}")
        print("-" * 40)
        item["details"] = details
        time.sleep(1)

    # 3) Save to CSV
    with open("shl_catalog_full.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "title", "remote_testing", "adaptive_irt", "test_type", "details"
        ])
        writer.writeheader()
        for item in catalog:
            writer.writerow(item)

    print("\nAll done! Data saved to shl_catalog_full.csv")

if __name__ == "__main__":
    main()

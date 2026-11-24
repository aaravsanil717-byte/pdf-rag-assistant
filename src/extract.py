import pymupdf4llm
import json


pages = pymupdf4llm.to_markdown("data/sample.pdf", page_chunks=True)

print(f"Success! Total pages extracted: {len(pages)}")
print("\n--- Preview of Page 1 ---\n")
print(pages[0]["text"][:1000])


with open("data/extracted_output.json", "w", encoding="utf-8") as f:
    json.dump(pages, f, indent=2, ensure_ascii=False)
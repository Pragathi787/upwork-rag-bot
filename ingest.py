from pypdf import PdfReader

pdf_path = "data/API Documentation Partial.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text()

print("Character count:")
print(len(text))

print("\nSample text:\n")
print(text[:1000])
import logging
from pdfparser import PDFTextExtractor

def main(log_level="INFO"):
  
  extractor = PDFTextExtractor()
  
  pdf_filepaths = [...]
  texts = extractor.extract_texts(pdf_filepaths)
  
  if not texts:
     raise ValueError("No text extracted from PDFs")
  
  # Process texts 
  ...

if __name__ == "__main__":
  logging.basicConfig(level=getattr(logging, log_level))  
  main()
import logging
import pdfminer

class PDFTextExtractor:
        
   def extract_texts(self, pdf_filepaths):
     
      texts = []  
        
      for pdf in pdf_filepaths:
         try:     
             text = self._extract_text(pdf) 
             texts.append(text)
         except pdfminer.errors.PDFReadError as e:  
             logging.warning(f"Error reading {pdf}: {e}")              
      return texts
         
   def _extract_text(self, pdf_filepath):

      "Extract text from a PDF file."   
     
       # PDFMiner code to extract text...
      
       return text
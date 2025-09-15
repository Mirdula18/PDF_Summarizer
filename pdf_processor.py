"""
PDF text extraction and processing functions
"""

import PyPDF2
import pdfplumber
import streamlit as st
from typing import Optional, List
import re

class PDFProcessor:
    def __init__(self):
        self.text_content = ""
        
    def extract_text_pypdf2(self, pdf_file) -> str:
        """
        Extract text from PDF using PyPDF2
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text + "\n"
                except Exception as e:
                    st.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                    continue
                    
            return text
        except Exception as e:
            st.error(f"Error reading PDF with PyPDF2: {str(e)}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_file) -> str:
        """
        Extract text from PDF using pdfplumber (better for complex layouts)
        """
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text + "\n"
                    except Exception as e:
                        st.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                        continue
            return text
        except Exception as e:
            st.error(f"Error reading PDF with pdfplumber: {str(e)}")
            return ""
    
    def extract_text(self, pdf_file, method="pdfplumber") -> str:
        """
        Main text extraction method with fallback
        """
        if method == "pdfplumber":
            text = self.extract_text_pdfplumber(pdf_file)
            if not text.strip():
                st.info("Pdfplumber extraction failed, trying PyPDF2...")
                pdf_file.seek(0)  # Reset file pointer
                text = self.extract_text_pypdf2(pdf_file)
        else:
            text = self.extract_text_pypdf2(pdf_file)
            if not text.strip():
                st.info("PyPDF2 extraction failed, trying pdfplumber...")
                pdf_file.seek(0)  # Reset file pointer
                text = self.extract_text_pdfplumber(pdf_file)
        
        return self.clean_text(text)
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers patterns (common ones)
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Remove URLs and email patterns
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def get_text_statistics(self, text: str) -> dict:
        """
        Get basic statistics about the extracted text
        """
        if not text:
            return {}
        
        sentences = text.split('.')
        words = text.split()
        characters = len(text)
        
        return {
            "characters": characters,
            "words": len(words),
            "sentences": len([s for s in sentences if s.strip()]),
            "pages_estimated": max(1, characters // 2000)  # Rough estimate
        }

"""
AI summarization logic using distilbart-cnn-12-6
"""

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict
import re
from config import *

class PDFSummarizer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.summarizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @st.cache_resource
    def load_model(_self):
        """
        Load the distilbart-cnn-12-6 model and tokenizer
        """
        try:
            with st.spinner("Loading AI model... This may take a few minutes on first run."):
                # Load tokenizer and model
                _self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
                _self.model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
                
                # Create summarization pipeline
                _self.summarizer = pipeline(
                    "summarization",
                    model=_self.model,
                    tokenizer=_self.tokenizer,
                    device=0 if _self.device == "cuda" else -1,
                    framework="pt"
                )
                
                st.success(f"✅ Model loaded successfully on {_self.device.upper()}")
                return True
                
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def split_text_into_chunks(self, text: str, max_length: int = MAX_CHUNK_LENGTH) -> List[str]:
        """
        Split text into chunks that fit the model's token limit
        """
        if not text:
            return []
        
        # Split by sentences first
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            # Rough token count estimation (1 token ≈ 0.75 words)
            estimated_tokens = len(potential_chunk.split()) * 1.3
            
            if estimated_tokens > max_length and current_chunk:
                # Add current chunk and start new one
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize_chunk(self, chunk: str, min_length: int = MIN_SUMMARY_LENGTH, 
                       max_length: int = MAX_SUMMARY_LENGTH) -> str:
        """
        Summarize a single chunk of text
        """
        try:
            if not chunk or len(chunk.split()) < 10:
                return ""
            
            # Adjust summary length based on input length
            input_length = len(chunk.split())
            adjusted_min = min(min_length, input_length // 3)
            adjusted_max = min(max_length, input_length // 2)
            
            summary = self.summarizer(
                chunk,
                min_length=max(adjusted_min, 10),
                max_length=min(adjusted_max, input_length - 5),
                do_sample=False,
                truncation=True
            )
            
            return summary[0]['summary_text'] if summary else ""
            
        except Exception as e:
            st.warning(f"Error summarizing chunk: {str(e)}")
            return ""
    
    def summarize_text(self, text: str, progress_bar=None) -> Dict[str, any]:
        """
        Main summarization function
        """
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            return {
                "success": False,
                "error": "Text is too short for summarization",
                "summary": "",
                "chunks_processed": 0,
                "original_length": len(text.split()),
                "summary_length": 0
            }
        
        # Load model if not already loaded
        if not self.summarizer:
            if not self.load_model():
                return {
                    "success": False,
                    "error": "Failed to load AI model",
                    "summary": "",
                    "chunks_processed": 0,
                    "original_length": len(text.split()),
                    "summary_length": 0
                }
        
        try:
            # Split text into manageable chunks
            chunks = self.split_text_into_chunks(text)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No valid text chunks found",
                    "summary": "",
                    "chunks_processed": 0,
                    "original_length": len(text.split()),
                    "summary_length": 0
                }
            
            summaries = []
            total_chunks = len(chunks)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                if progress_bar:
                    progress_bar.progress((i + 1) / total_chunks)
                
                chunk_summary = self.summarize_chunk(chunk)
                if chunk_summary:
                    summaries.append(chunk_summary)
            
            # Combine summaries
            final_summary = " ".join(summaries)
            
            # If we have multiple summaries, create a final condensed version
            if len(summaries) > 1 and len(final_summary.split()) > MAX_SUMMARY_LENGTH:
                final_summary = self.summarize_chunk(
                    final_summary, 
                    min_length=MIN_SUMMARY_LENGTH,
                    max_length=MAX_SUMMARY_LENGTH
                )
            
            return {
                "success": True,
                "summary": final_summary,
                "chunks_processed": len(summaries),
                "original_length": len(text.split()),
                "summary_length": len(final_summary.split()),
                "compression_ratio": round(len(final_summary.split()) / len(text.split()) * 100, 2)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Summarization failed: {str(e)}",
                "summary": "",
                "chunks_processed": 0,
                "original_length": len(text.split()),
                "summary_length": 0
            }

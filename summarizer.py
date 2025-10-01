"""
AI summarization logic using distilbart-cnn-12-6
"""

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any, Optional, Union
from config import *
from performance import Timer, PerformanceMonitor, time_operation
import time

class PDFSummarizer:
    def __init__(self):
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.summarizer: Optional[Any] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded: bool = False
        self.performance_monitor = PerformanceMonitor()
        
    @time_operation("Model Loading")
    def load_model(self) -> bool:
        """
        Load the distilbart-cnn-12-6 model and tokenizer with optimized caching
        """
        # Quick check if model is already loaded and functional
        if self.model_loaded and self.summarizer is not None:
            return True
        
        # Check if model exists but needs verification
        if self.summarizer is not None and not self.model_loaded:
            try:
                # Quick functional test without full reload
                test_result = self.summarizer("Test.", max_length=30, min_length=5, do_sample=False)
                if test_result and len(test_result) > 0:
                    self.model_loaded = True
                    return True
            except Exception:
                # Model is not working, need to reload
                pass
        
        try:
            with Timer("Model initialization", display=True):
                with st.spinner("🤖 Loading AI model (this may take a moment on first run)..."):
                    # Load tokenizer and model with optimizations
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        MODEL_NAME,
                        cache_dir=None,  # Use default cache
                        local_files_only=False
                    )
                    
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        MODEL_NAME,
                        cache_dir=None,  # Use default cache
                        local_files_only=False,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,  # Use half precision on GPU
                        low_cpu_mem_usage=True  # Optimize memory usage
                    )
                    
                    # Move model to device
                    if self.device == "cuda":
                        self.model = self.model.cuda()
                    
                    # Create optimized pipeline
                    self.summarizer = pipeline(
                        "summarization",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=0 if self.device == "cuda" else -1,
                        framework="pt",
                        batch_size=8 if self.device == "cuda" else 4  # Optimize batch size
                    )
                    
                    self.model_loaded = True
                    st.success(f"✅ Model loaded successfully on {self.device.upper()}!")
                    return True
                
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            self.summarizer = None
            self.model_loaded = False
            return False
    
    def split_text_into_chunks(self, text: str, max_length: int = MAX_CHUNK_LENGTH) -> List[str]:
        """
        Optimized text chunking with better token estimation
        """
        if not text or len(text.strip()) < 50:
            return []
        
        # More efficient sentence splitting using multiple delimiters
        import re
        sentences = re.split(r'[.!?]+\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        chunks: List[str] = []
        current_chunk: str = ""
        
        # Better token estimation using tokenizer if available
        def estimate_tokens(text: str) -> int:
            if self.tokenizer:
                try:
                    return len(self.tokenizer.encode(text, add_special_tokens=False))
                except:
                    pass
            # Fallback to word-based estimation
            return int(len(text.split()) * 1.3)
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            estimated_tokens = estimate_tokens(potential_chunk)
            
            if estimated_tokens > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very small chunks and merge them
        filtered_chunks = []
        for chunk in chunks:
            if len(chunk.split()) < 20 and filtered_chunks:
                # Merge small chunk with previous one if it won't exceed limit
                potential_merge = filtered_chunks[-1] + " " + chunk
                if estimate_tokens(potential_merge) <= max_length:
                    filtered_chunks[-1] = potential_merge
                else:
                    filtered_chunks.append(chunk)
            else:
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def summarize_chunk(self, chunk: str, min_length: int = None, 
                       max_length: int = None) -> str:
        """
        Summarize a single chunk of text with optimizations
        """
        try:
            if not chunk or len(chunk.split()) < 10:
                return ""
            
            # Ensure summarizer is loaded and functional
            if not self.model_loaded:
                if not self.load_model():
                    return ""
            
            # Use provided parameters or fall back to config defaults
            if min_length is None:
                min_length = MIN_SUMMARY_LENGTH
            if max_length is None:
                max_length = MAX_SUMMARY_LENGTH
            
            # Adjust summary length based on input length
            input_length: int = len(chunk.split())
            adjusted_min: int = min(min_length, max(input_length // 4, 5))
            adjusted_max: int = min(max_length, max(input_length // 2, adjusted_min + 10))
            
            # Optimized summarization with better parameters
            summary = self.summarizer(
                chunk,
                min_length=adjusted_min,
                max_length=adjusted_max,
                do_sample=False,
                truncation=True,
                clean_up_tokenization_spaces=True,
                early_stopping=True,  # Stop when good summary is found
                no_repeat_ngram_size=2  # Reduce repetition
            )
            
            return summary[0]['summary_text'].strip() if summary and len(summary) > 0 else ""
            
        except Exception as e:
            st.warning(f"⚠️ Error summarizing chunk: {str(e)}")
            return ""
    
    def summarize_chunks_batch(self, chunks: List[str], min_length: int = None, 
                              max_length: int = None) -> List[str]:
        """
        Batch process multiple chunks for better performance
        """
        if not chunks:
            return []
        
        # Ensure model is loaded
        if not self.model_loaded:
            if not self.load_model():
                return []
        
        # Use provided parameters or fall back to config defaults
        if min_length is None:
            min_length = MIN_SUMMARY_LENGTH
        if max_length is None:
            max_length = MAX_SUMMARY_LENGTH
        
        summaries = []
        batch_size = 4 if self.device == "cuda" else 2  # Adjust based on available memory
        
        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_summaries = []
                
                for chunk in batch:
                    if len(chunk.split()) < 10:
                        batch_summaries.append("")
                        continue
                    
                    input_length = len(chunk.split())
                    adjusted_min = min(min_length, max(input_length // 4, 5))
                    adjusted_max = min(max_length, max(input_length // 2, adjusted_min + 10))
                    
                    try:
                        result = self.summarizer(
                            chunk,
                            min_length=adjusted_min,
                            max_length=adjusted_max,
                            do_sample=False,
                            truncation=True,
                            clean_up_tokenization_spaces=True,
                            early_stopping=True,
                            no_repeat_ngram_size=2
                        )
                        
                        summary_text = result[0]['summary_text'].strip() if result and len(result) > 0 else ""
                        batch_summaries.append(summary_text)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error in batch processing: {str(e)}")
                        batch_summaries.append("")
                
                summaries.extend(batch_summaries)
                
                # Small delay to prevent overwhelming the GPU
                if self.device == "cuda" and len(chunks) > batch_size:
                    time.sleep(0.1)
        
        except Exception as e:
            st.error(f"❌ Batch processing failed: {str(e)}")
            # Fallback to individual processing
            return [self.summarize_chunk(chunk, min_length, max_length) for chunk in chunks]
        
        return summaries
    
    @time_operation("Text Summarization")
    def summarize_text(self, text: str, progress_bar: Optional[Any] = None, 
                      min_length: int = None, max_length: int = None) -> Dict[str, Union[bool, str, int, float, None]]:
        """
        Optimized main summarization function with performance monitoring
        """
        start_time = time.time()
        self.performance_monitor = PerformanceMonitor()  # Reset metrics
        
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            return {
                'success': False,
                'error': 'Text is too short to summarize',
                'summary': '',
                'chunks_processed': 0,
                'total_chunks': 0,
                'original_length': len(text.split()),
                'summary_length': 0,
                'compression_ratio': 0.0,
                'total_time': 0.0
            }
        
        # Load model with timing
        model_load_start = time.time()
        if not self.model_loaded:
            if not self.load_model():
                return {
                    'success': False,
                    'error': 'Failed to load AI model',
                    'summary': '',
                    'chunks_processed': 0,
                    'total_chunks': 0,
                    'original_length': len(text.split()),
                    'summary_length': 0,
                    'compression_ratio': 0.0,
                    'total_time': time.time() - start_time
                }
        
        model_load_time = time.time() - model_load_start
        self.performance_monitor.record_metric('model_loading', model_load_time)
        
        try:
            # Split text into manageable chunks with timing
            chunk_start = time.time()
            chunks: List[str] = self.split_text_into_chunks(text)
            chunk_time = time.time() - chunk_start
            self.performance_monitor.record_metric('text_chunking', chunk_time)
            
            total_chunks: int = len(chunks)
            
            if total_chunks == 0:
                return {
                    'success': False,
                    'error': 'No valid text chunks found',
                    'summary': '',
                    'chunks_processed': 0,
                    'total_chunks': 0,
                    'original_length': len(text.split()),
                    'summary_length': 0,
                    'compression_ratio': 0.0,
                    'total_time': time.time() - start_time
                }
            
            # Use batch processing for better performance
            summarization_start = time.time()
            
            if total_chunks <= 10:  # Use batch processing for smaller sets
                summaries = self.summarize_chunks_batch(chunks, min_length, max_length)
                chunks_processed = sum(1 for s in summaries if s.strip())
            else:  # Process individually with progress updates for larger sets
                summaries: List[str] = []
                chunks_processed: int = 0
                
                for i, chunk in enumerate(chunks):
                    if progress_bar:
                        progress_bar.progress((i + 1) / total_chunks)
                    
                    chunk_summary: str = self.summarize_chunk(chunk, min_length, max_length)
                    if chunk_summary:
                        summaries.append(chunk_summary)
                        chunks_processed += 1
                    else:
                        summaries.append("")
            
            summarization_time = time.time() - summarization_start
            self.performance_monitor.record_metric('summarization', summarization_time)
            
            # Filter out empty summaries
            valid_summaries = [s for s in summaries if s.strip()]
            
            if not valid_summaries:
                return {
                    'success': False,
                    'error': 'No summaries could be generated',
                    'summary': '',
                    'chunks_processed': 0,
                    'total_chunks': total_chunks,
                    'original_length': len(text.split()),
                    'summary_length': 0,
                    'compression_ratio': 0.0,
                    'total_time': time.time() - start_time
                }
            
            # Combine summaries
            final_summary: str = " ".join(valid_summaries)
            
            # If we have multiple chunks, summarize the combined summaries
            final_max_length = max_length if max_length is not None else MAX_SUMMARY_LENGTH
            if len(valid_summaries) > 1 and len(final_summary.split()) > final_max_length:
                final_summary = self.summarize_chunk(
                    final_summary, 
                    min_length=min_length,
                    max_length=max_length
                )
            
            # Calculate statistics
            total_time = time.time() - start_time
            original_length = len(text.split())
            summary_length = len(final_summary.split())
            compression_ratio = round((summary_length / original_length) * 100, 2) if original_length > 0 else 0.0
            
            # Record total processing time
            self.performance_monitor.record_metric('total_processing', total_time)
            
            return {
                'success': True,
                'summary': final_summary,
                'chunks_processed': chunks_processed,
                'total_chunks': total_chunks,
                'original_length': original_length,
                'summary_length': summary_length,
                'compression_ratio': compression_ratio,
                'total_time': total_time,
                'performance_metrics': self.performance_monitor.metrics,
                'error': None
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            return {
                'success': False,
                'error': f'Summarization failed: {str(e)}',
                'summary': '',
                'chunks_processed': 0,
                'total_chunks': 0,
                'original_length': len(text.split()),
                'summary_length': 0,
                'compression_ratio': 0.0,
                'total_time': total_time,
                'performance_metrics': {}
            }

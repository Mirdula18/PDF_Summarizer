"""
Main Streamlit application for PDF Summarizer
"""

import streamlit as st
import time
from pdf_processor import PDFProcessor
from summarizer import PDFSummarizer
from utils import *
from config import *

def main():
    # Set up page
    setup_page_config()
    
    # Header
    st.title("📄 PDF Summarizer")
    st.markdown("### Transform lengthy PDFs into concise summaries using AI")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("🤖 AI Model")
        display_model_info()
        
        st.header("⚙️ Settings")
        extraction_method = st.selectbox(
            "PDF Extraction Method",
            ["pdfplumber", "PyPDF2"],
            help="pdfplumber works better with complex layouts"
        )
        
        summary_length = st.select_slider(
            "Summary Length",
            options=["Short", "Medium", "Long"],
            value="Medium",
            help="Adjust the length of generated summaries"
        )
    
    # Adjust summary parameters based on user choice
    if summary_length == "Short":
        min_len, max_len = 30, 100
    elif summary_length == "Long":
        min_len, max_len = 100, 300
    else:
        min_len, max_len = MIN_SUMMARY_LENGTH, MAX_SUMMARY_LENGTH
    
    # Main content
    st.header("📤 Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file:
        # Validate file
        validation = validate_file(uploaded_file)
        if not validation['valid']:
            st.error(validation['error'])
            return
        
        st.success(f"✅ File uploaded successfully ({validation['size_mb']}MB)")
        
        # Process PDF button
        if st.button("🚀 Extract Text & Generate Summary", type="primary"):
            
            # Step 1: Extract text
            with st.spinner("📖 Extracting text from PDF..."):
                processor = PDFProcessor()
                extracted_text = processor.extract_text(uploaded_file, method=extraction_method)
            
            if not extracted_text:
                st.error("❌ Could not extract text from PDF. Please try a different file or extraction method.")
                return
            
            # Display text statistics
            st.subheader("📊 Document Statistics")
            stats = processor.get_text_statistics(extracted_text)
            display_text_stats(stats)
            
            # Show preview of extracted text
            with st.expander("👀 Preview Extracted Text"):
                st.text_area(
                    "First 1000 characters:",
                    extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
                    height=200,
                    disabled=True
                )
            
            # Step 2: Generate summary
            st.subheader("🤖 AI Summary Generation")
            
            summarizer = PDFSummarizer()
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🧠 Generating AI summary..."):
                status_text.text("Initializing AI model...")
                
                # Update MIN/MAX lengths for this session
                import config
                config.MIN_SUMMARY_LENGTH = min_len
                config.MAX_SUMMARY_LENGTH = max_len
                
                result = summarizer.summarize_text(extracted_text, progress_bar)
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.subheader("📋 Results")
            display_summary_results(result)
            
            if result.get('success'):
                # Option to download detailed report
                detailed_report = create_summary_report(
                    extracted_text, 
                    result, 
                    uploaded_file.name
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📊 Download Detailed Report",
                        data=detailed_report,
                        file_name=f"{uploaded_file.name}_summary_report.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    # Option to summarize further if needed
                    if st.button("🔄 Regenerate Summary"):
                        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit and HuggingFace Transformers</p>
            <p><em>Powered by distilbart-cnn-12-6 AI model</em></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

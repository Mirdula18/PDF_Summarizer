import streamlit as st
from typing import Dict, Any
from pdf_processor import PDFProcessor
from summarizer import PDFSummarizer
from utils import *
from config import *

def main():
    # Set up page
    setup_page_config()
    
    # Header
    st.title("PDF Summarizer")
    st.markdown("### Transform lengthy PDFs into concise summaries using AI")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("AI Model")
        display_model_info()
        
        st.header("Settings")
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
    st.header("Upload PDF")
    
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
        
        st.success(f"File uploaded successfully ({validation['size_mb']}MB)")
        
        # Process PDF button
        if st.button("🚀 Extract Text & Generate Summary", type="primary"):
            
            # Step 1: Extract text
            with st.spinner("Extracting text from PDF..."):
                processor = PDFProcessor()
                extracted_text = processor.extract_text(uploaded_file, method=extraction_method)
            
            if not extracted_text:
                st.error("Could not extract text from PDF. Please try a different file or extraction method.")
                return
            
            # Display text statistics
            st.subheader("Document Statistics")
            stats = processor.get_text_statistics(extracted_text)
            display_text_stats(stats)
            
            # Show preview of extracted text
            with st.expander("Preview Extracted Text"):
                st.text_area(
                    "First 1000 characters:",
                    extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
                    height=200,
                    disabled=True
                )
            
            # Step 2: Generate summary
            st.subheader("AI Summary Generation")
            
            # Initialize or get existing summarizer from session state
            if 'summarizer' not in st.session_state:
                st.session_state.summarizer = PDFSummarizer()
            
            summarizer = st.session_state.summarizer
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Generating AI summary..."):
                status_text.text("Initializing AI model...")
                
                # Pass the selected min/max lengths directly to the summarizer
                result = summarizer.summarize_text(extracted_text, progress_bar, min_len, max_len)
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.subheader("Results")
            display_summary_results(result)
            
            # Display performance metrics if available
            if result.get('success') and 'performance_metrics' in result:
                st.subheader("Performance Metrics")
                metrics = result['performance_metrics']
                
                if metrics:
                    cols = st.columns(len(metrics))
                    for i, (name, data) in enumerate(metrics.items()):
                        with cols[i]:
                            st.metric(
                                label=name.replace('_', ' ').title(),
                                value=f"{data['value']:.2f}s"
                            )
                
                # Show total time prominently
                if 'total_time' in result:
                    st.info(f"**Total Processing Time: {result['total_time']:.2f} seconds**")
            
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
                        label="Download Detailed Report",
                        data=detailed_report,
                        file_name=f"{uploaded_file.name}_summary_report.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    # Option to summarize further if needed
                    if st.button("Regenerate Summary"):
                        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built using Streamlit and HuggingFace Transformers</p>
            <p><em>Powered by distilbart-cnn-12-6 AI model</em></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

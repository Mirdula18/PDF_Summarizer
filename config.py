# Model Configuration
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
MAX_CHUNK_LENGTH = 1024  # Maximum tokens per chunk for the model
MIN_SUMMARY_LENGTH = 50
MAX_SUMMARY_LENGTH = 200

# PDF Processing Configuration
MAX_FILE_SIZE_MB = 10
SUPPORTED_FORMATS = ['.pdf']

# Streamlit Configuration
PAGE_TITLE = "PDF Summarizer"
PAGE_ICON = "📄"
LAYOUT = "wide"

# Text Processing Configuration
SENTENCE_OVERLAP = 2  # Number of sentences to overlap between chunks
MIN_TEXT_LENGTH = 100  # Minimum text length to process

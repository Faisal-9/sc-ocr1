# ==========================================
# GENERAL SETTINGS
# ==========================================

TEMP_FOLDER = "temp"
DEFAULT_LANG = "rus+eng"

# ==========================================
# GPU SETTINGS
# ==========================================

USE_GPU = True

PRIMARY_OCR_ENGINE = "paddle" 

USE_EASYOCR_FALLBACK = True
USE_TESSERACT_FALLBACK = True

# ==========================================
# OCR SETTINGS
# ==========================================

UPSCALE_FACTOR = 4
TABLE_UPSCALE_FACTOR = 2
OCR_ZOOM = 5.0
OCR_BATCH_SIZE = 1

ENABLE_ENSEMBLE = True
ENABLE_TEXT_LAYER_FOR_PDF = True
ENABLE_RUSSIAN_CORRECTION = False
ENABLE_LAYOUT_RECONSTRUCTION = True
ENABLE_TABLE_EXTRACTION = True
ENABLE_IMAGE_EXTRACTION = True
ENABLE_GRAPH_EXTRACTION = True

# ==========================================
# TABLE SETTINGS
# ==========================================

TABLE_DETECTION_MODE = "hybrid"
ENABLE_BORDERLESS_TABLES = True
MAX_TABLES_PER_PAGE = 20

# ==========================================
# IMAGE SETTINGS
# ==========================================

SAVE_ORIGINAL_IMAGES = True
SAVE_EXTRACTED_IMAGES = True
IMAGE_OUTPUT_FORMAT = "png"

# ==========================================
# PREPROCESSING
# ==========================================

ENABLE_DESKEW = True
ENABLE_CROP = True
ENABLE_DENOISE = True
ENABLE_CLAHE = True
ENABLE_SHARPEN = True
ENABLE_SUPER_RESOLUTION = True

# ==========================================
# COMPONENT FILTERING
# ==========================================

MIN_COMPONENT_AREA = 500

# ==========================================
# OCR QUALITY CONTROL
# ==========================================

MIN_ACCEPTABLE_CONFIDENCE = 0.75
REVIEW_THRESHOLD = 0.80
HIGH_CONFIDENCE_THRESHOLD = 0.90

# ==========================================
# PDF PROCESSING
# ==========================================

PDF_RENDER_DPI = 500
PDF_RENDER_ZOOM = 5.0

# ==========================================
# PERFORMANCE
# ==========================================

MAX_WORKERS = 1
ENABLE_CACHE = True

# ==========================================
# STREAMLIT
# ==========================================

SHOW_DEBUG_PANEL = True
SHOW_GPU_STATUS = True
SHOW_PROCESSING_TIMER = True
SHOW_ETA = True
SHOW_PAGE_PROGRESS = True
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NFS_BASE = "/mnt/nfs/hndb"  # Base path for NFS storage

# Static files
STATIC_DIR = "static"

# Sample directories
SAMPLE_FILES_DIR = os.path.join(NFS_BASE, "Sample_Files")
SAMPLE_TEMP_DIR = os.path.join(NFS_BASE, "temp", "Sample_temp")
SAMPLE_SNAPSHOT_DIR_TEMPLATE = os.path.join(SAMPLE_FILES_DIR, "{idx}_{folder}/sample_snapshot")
SAMPLE_IMAGE_DIR_TEMPLATE = os.path.join(SAMPLE_FILES_DIR, "{idx}_{folder}/sample_image")
SAMPLE_ANNOTATION_DIR_TEMPLATE = os.path.join(SAMPLE_FILES_DIR, "{idx}_{folder}/sample_annoation")

# Image processing directories
V3DRAW_16BIT_DIR = os.path.join(NFS_BASE, "V3DRAW_16bit")
V3DRAW_8BIT_DIR = os.path.join(NFS_BASE, "V3DRAW_8bit")
MIP_DOWNSAMPLE_DIR = os.path.join(NFS_BASE, "MIP_Downsample")
V3DPBD_DIR = os.path.join(NFS_BASE, "V3DPBD")
RAW_2D_IMAGES_DIR = os.path.join(NFS_BASE, "2D_raw_images")

# Sample preparation
SAMPLE_PREPARATION_DIR = os.path.join(NFS_BASE, "SamplePreparation")
SAMPLE_PREPARATION_parent_DIR = os.path.join(SAMPLE_PREPARATION_DIR,"{sample_id}")
SAMPLE_PREPARATION_DIR_TEMPLATE = os.path.join(SAMPLE_PREPARATION_DIR, "{sample_id}/{sample_id}")
SAMPLE_PREPARATION_IMAGING_TEMPLATE = os.path.join(SAMPLE_PREPARATION_DIR, "{sample_id}/{sample_id}-{imaging_id}")

# Temp and upload directories
TEMP_DIR = os.path.join(NFS_BASE, "temp")
TEMP_XLSX_PATH = os.path.join(TEMP_DIR, "sample_data.xlsx")
TEMP_MARKER_TEMPLATE = os.path.join(TEMP_DIR, "{cell_id}.marker")
TEMP_ZIP_TEMPLATE = os.path.join(TEMP_DIR, "{cell_id}.zip")

# Record book pictures
RECORD_BOOK_PICS_DIR = os.path.join(NFS_BASE, "Record_Book_Pics")

# Injection files
INJECTION_FILES_BASE = os.path.join(NFS_BASE, "Injection_Files")
INJECTION_ORIGINAL_DIR = os.path.join(INJECTION_FILES_BASE, "Original")
INJECTION_TEMP_DIR = os.path.join(INJECTION_FILES_BASE, "temp")
INJECTION_DB_UPLOAD_DIR = os.path.join(INJECTION_FILES_BASE, "DB_Uploads")

# Imaging files
IMAGING_FILES_BASE = os.path.join(NFS_BASE, "Imaging_Files")
IMAGING_METADATA_DIR = os.path.join(IMAGING_FILES_BASE, "Metadata")
MARKER_FILES_DIR = os.path.join(IMAGING_FILES_BASE, "Markers")
ANNOTATION_FILES_DIR = os.path.join(IMAGING_FILES_BASE, "Annotations")
IMAGING_MATCHTABLE_DIR = os.path.join(IMAGING_FILES_BASE, "MatchTables")

# Image DB directory
PTRSB_DB_DIR = "/PB/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/PTRSB_DB"

# Vaa3D
VAA3D_PATH = "/vaa3d/Vaa3D-x.1.1.4Ubuntu/Vaa3D-x"

# Helper function to ensure directories exist
def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path
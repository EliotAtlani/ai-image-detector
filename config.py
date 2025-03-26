import os
# Global configuration parameters
IMG_SIZE = (250, 250)
DATA_DIR = "../data/art"
TFRECORDS_FILE = "dataset.tfrecords"
OUTPUT_DIR = "../data_resized/art"
OUTPUT_TFRECORD_DIR = "../data_processed/art"
# Derived paths for class-specific directories
FAKE_DIR = os.path.join(DATA_DIR, "fake")
REAL_DIR = os.path.join(DATA_DIR, "real")

import logging
import os
from datetime import datetime

LOG_FORMAT = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"

# Serverless filesystems (Vercel) are read-only, so log to stderr there and
# keep timestamped log files for local runs.
if os.environ.get("VERCEL"):
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
else:
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)
    LOG_FILE_PATH = os.path.join(logs_path, f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log")
    logging.basicConfig(filename=LOG_FILE_PATH, format=LOG_FORMAT, level=logging.INFO)

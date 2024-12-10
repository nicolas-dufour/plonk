import os, zipfile
from huggingface_hub import snapshot_download

# Define the base directory
base_dir = os.path.join(os.getcwd(), 'datasets')

# Ensure the base directory exists
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Define the specific dataset directory
dataset_dir = os.path.join(base_dir, "osv5m")

# Ensure the specific dataset directory exists
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)

# Download the dataset
snapshot_download(repo_id="osv5m/osv5m", local_dir=dataset_dir, repo_type='dataset')

# Extract zip files and remove them after extraction
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                zip_ref.extractall(root)
                os.remove(os.path.join(root, file))

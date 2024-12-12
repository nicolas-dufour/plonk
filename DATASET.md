### Dataset
To download the datataset, run:
```python
# download the full dataset
from huggingface_hub import snapshot_download
snapshot_download(repo_id="osv5m/osv5m", local_dir="datasets/osv5m", repo_type='dataset')
```

and finally extract:
```python
import os
import zipfile
for root, dirs, files in os.walk("datasets/osv5m"):
    for file in files:
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                zip_ref.extractall(root)
                os.remove(os.path.join(root, file))
```

You can also directly load the dataset using `load_dataset`:
```python
from plonk.datasets import load_dataset
dataset = load_dataset('osv5m/osv5m', full=False)
```
where with `full` you can specify whether you want to load the complete metadata (default: `False`).

If you only want to download the test set, you can run the script below:
```python
from huggingface_hub import hf_hub_download
for i in range(5):
    hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/test", repo_type='dataset', local_dir="datasets/osv5m")
    hf_hub_download(repo_id="osv5m/osv5m", filename="README.md", repo_type='dataset', local_dir="datasets/osv5m")
```
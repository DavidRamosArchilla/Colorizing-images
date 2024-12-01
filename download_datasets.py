import os 
import requests
import zipfile
from tqdm import tqdm

def download_coco(root_dir, split='train'):
    url = "http://images.cocodataset.org/zips/train2017.zip" if split == 'train' else "http://images.cocodataset.org/zips/val2017.zip" # ES EL CONJUNTO DE VALIDACION, COMO PRIMERA PRUEBA PUEDE ESTAR BIEN PERO HABRIA QEU CAMBIARLO # "http://images.cocodataset.org/zips/train2017.zip"
    filename = os.path.join(root_dir, f"{split}2017.zip")

    if not os.path.exists(filename):
        print("Downloading COCO 2017 validation set...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filename, "wb") as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

    # Extract the zip file
    if not os.path.exists(os.path.join(root_dir, f"{split}2017")):
        print("Extracting files...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(root_dir)

download_coco('data', 'val')
download_coco('data', 'train')
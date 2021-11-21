import os
import glob
import subprocess
import pathlib
import time
import shutil

from tqdm import tqdm

ravdess_extracted_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'RAVDESS_extracted'))
ravdess_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'RAVDESS'))

if os.path.isdir(ravdess_extracted_path):
    shutil.rmtree(ravdess_extracted_path)
pathlib.Path(ravdess_extracted_path).mkdir(parents=True, exist_ok=True)

os.chdir(ravdess_extracted_path)
zip_paths = list(pathlib.Path(ravdess_path).glob('*.zip'))
for path in tqdm(zip_paths):
    subprocess.run(f'unzip {path}', check=True, shell=True, stdout=open(os.devnull, 'w'))

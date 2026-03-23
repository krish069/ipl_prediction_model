import os
import requests
import time
import zipfile
import io
import subprocess

cricsheet = "https://cricsheet.org/downloads/ipl_male_json.zip"
json_folder = "ipl_data"

def download_and_extract_zip():
    response = requests.get(cricsheet)

    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(json_folder)
        print("\nData downloaded and extracted successfully.\n")
        return True
    else:
        print(f"\nFailed to download data. Status code: {response.status_code}\n")
        return False
    
def run_pipeline():
    scripts = [
        'parsing.py',
        'calculate_elo.py',
        'calculate_player_stats.py',
        'calculate_venue_stats.py',
        'train_model.py',
    ]
    for script in scripts:
        print(f"\nRunning {script}...")
        result = subprocess.run(['python', script])
        if result.returncode != 0:
            print(f"Pipeline stopped — {script} failed with return code {result.returncode}.")
            return
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    if download_and_extract_zip():
        run_pipeline()

        start_time = time.time()

        elapsed_time = time.time() - start_time
        print(f"\nPipeline executed in {elapsed_time:.2f} seconds.\n")
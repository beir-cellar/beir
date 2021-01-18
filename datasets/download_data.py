import os
import requests 
import zipfile
import pathlib

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def main():
    
    dataset_files = ["nfcorpus.zip", "fiqa.zip", "dbpedia-entity.zip", "hotpotqa.zip", "newsqa.zip", "trec-covid.zip", "webis-touche2020.zip"]
    out_dir = pathlib.Path(__file__).parent.absolute()
    
    for dataset in dataset_files:
        zip_file = os.path.join(out_dir, dataset)
        
        if not os.path.isfile(zip_file):
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}".format(dataset)
            print("Downloading {} ...".format(dataset))
            download_url(url, zip_file)
        
        if not os.path.isdir(zip_file.replace(".zip", "")):
            print("Extracting {} ...".format(dataset))
            zip_ = zipfile.ZipFile(zip_file, "r")
            zip_.extractall(path=out_dir)
            zip_.close()

if __name__ == '__main__':
    main()
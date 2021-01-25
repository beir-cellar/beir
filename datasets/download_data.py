import os
import pathlib

import beir.util

def main():
    
    out_dir = pathlib.Path(__file__).parent.absolute()
    
    dataset_files = ["nfcorpus.zip", "fiqa.zip", "dbpedia-entity.zip", 
                     "hotpotqa.zip", "newsqa.zip", "trec-covid.zip", 
                     "webis-touche2020.zip", "climate-fever.zip", 
                     "fever.zip", "cqadupstack.zip"]
    
    for dataset in dataset_files:
        
        zip_file = os.path.join(out_dir, dataset)
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}".format(dataset)
        
        print("Downloading {} ...".format(dataset))
        beir.util.download_url(url, zip_file)
        
        print("Unzipping {} ...".format(dataset))
        beir.util.unzip(zip_file, out_dir)

if __name__ == '__main__':
    main()
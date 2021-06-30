import os
import pathlib
from beir import util

def main():
    
    out_dir = pathlib.Path(__file__).parent.absolute()
    
    dataset_files = ["msmarco.zip", "trec-covid.zip", "nfcorpus.zip",
                     "nq.zip", "hotpotqa.zip", "fiqa.zip", "arguana.zip",
                     "webis-touche2020.zip", "cqadupstack.zip", "quora.zip",
                     "dbpedia-entity.zip", "scidocs.zip", "fever.zip",
                     "climate-fever.zip", "scifact.zip", "germanquad.zip"]
    
    for dataset in dataset_files:
        
        zip_file = os.path.join(out_dir, dataset)
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}".format(dataset)
        
        print("Downloading {} ...".format(dataset))
        util.download_url(url, zip_file)
        
        print("Unzipping {} ...".format(dataset))
        util.unzip(zip_file, out_dir)

if __name__ == '__main__':
    main()
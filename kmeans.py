import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm.auto import trange
import argparse
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import manifold

def main(args):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    dataset = load_dataset(args.dataset, streaming=False, split="train")
    iter_dataset = iter(dataset)
    embeddings = torch.ones(1, 384)
    def get_batch(s=8):
        b = [next(iter_dataset)['question'] for i in range(s)]
        return  b
    for i in trange(args.initial_samples//args.initial_batch_size):
        embeddings = torch.cat([embeddings, torch.tensor(model.encode(get_batch(args.initial_batch_size)))])
    kmeans = KMeans(n_clusters=args.clusters).fit(embeddings[1:])
    centers = torch.tensor(kmeans.cluster_centers_)
    torch.save(centers, args.cluster_file)
    def cluster(batch):
        embedded = torch.tensor(model.encode(batch[args.dataset_field]))
        cluster = [(i.unsqueeze(0) - centers).pow(2).mean(1).argmin().item() for i in embedded]
        return {
            'cluster': cluster
        }
    new_data = dataset.map(lambda batch: cluster(batch), batched=True)
    new_data.push_to_hub(args.output_repo)
    
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--initial-samples", default=16384)
parser.add_argument("--initial-batch-size", default=8)
parser.add_argument("--clusters", default=8)
parser.add_argument("--cluster-file", default="clusters.pt")
parser.add_argument("--dataset-field", default="text")
parser.add_argument("--output-repo")


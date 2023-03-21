
import wandb
import argparse
from dotmap import DotMap
from data import *
from train import *

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--root_path', type=str, help='Root directory', default='/data/dahn017/KDD23')
    parser.add_argument('--dataset', type=str, help='Dataset: fakeddit, politifact, gossipcop, kaggle', default='politifact')

    # Pretrained model choice
    parser.add_argument('--llm', type=str, help='Language model types: sbert, clip, and bart', default='clip')
    parser.add_argument('--lvm', type=str, help='Vision model types: clip, Vit, and ResNet', default='clip')

    # Fusing method
    parser.add_argument('--fuse', type=str, help='Fusing embedding types: concat, tensor (symcp)', default='concat')

    # Hyper parameters
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--dropout', type=float, default = 0.8)
    parser.add_argument('--act', type=str, default = 'ReLU')
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=256)
    
    # Classifier
    parser.add_argument('--clf', type=str, help='Classifier types: skl_logress, pt_mlp', default='skl_logress')

    args = parser.parse_args()

    dict_args = DotMap(dict(args._get_kwargs()))
    dict_args.dropout = float(dict_args.dropout)
    dict_args.device = 'cuda:2'

    return dict_args


def main():

    config = parse_args()
    wandb.init(
        project='CS260 - Alignment',
        notes = 'Alignment experiments (GOT) - single multi-modal (one text and one image)',
        config = config
    )

    dataset = dataset_loader(config)
    results = train(dataset, config)
    wandb.log(results)

if __name__ ==  '__main__':
    main()

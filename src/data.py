
import torch
import numpy as np
from dotmap import DotMap


def dataset_loader(config):

    data_dict = DotMap()
    device = config.device
    root_path = config.root_path
    dataset = config.dataset
    llm = config.llm
    lvm = config.lvm
    data_path = f'{root_path}/datasets/{dataset}'

    text_embed = np.load(f'{data_path}/pretrained/text_{llm}_embeds.npy')
    img_embed = np.load(f'{data_path}/pretrained/text_{lvm}_embeds.npy')

    y_train = np.load(f'{data_path}/y_train.npy')
    y_valid = np.load(f'{data_path}/y_valid.npy')
    y_test = np.load(f'{data_path}/y_test.npy')
    train_indices = np.load(f"{data_path}/train_idxs.npy")
    valid_indices = np.load(f"{data_path}/valid_idxs.npy")
    test_indices = np.load(f"{data_path}/test_idxs.npy")

    train_text_embed = np.nan_to_num(text_embed)[train_indices]
    test_text_embed = np.nan_to_num(text_embed)[test_indices]
    train_img_embed = np.nan_to_num(img_embed)[train_indices]
    test_img_embed = np.nan_to_num(img_embed)[test_indices]

    data_dict.text_embed = torch.FloatTensor(np.nan_to_num(text_embed)).to(device)
    data_dict.img_embed = torch.FloatTensor(np.nan_to_num(img_embed)).to(device)
    data_dict.train_text_embed = torch.FloatTensor(train_text_embed).to(device)
    data_dict.test_text_embed = torch.FloatTensor(test_text_embed)
    data_dict.train_img_embed = torch.FloatTensor(train_img_embed).to(device)
    data_dict.test_img_embed = torch.FloatTensor(test_img_embed)
    data_dict.y_train = y_train
    data_dict.y_valid = y_valid
    data_dict.y_test = y_test
    data_dict.train_indices = train_indices
    data_dict.valid_indices = valid_indices
    data_dict.test_indices = test_indices

    return data_dict


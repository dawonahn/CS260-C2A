
import torch
import torch.optim as optim
from clf import *
from fc import *
from ot import *

def train(dataset, config):

	device = config.device
	text_embed = dataset.text_embed
	img_embed = dataset.img_embed
	train_text_embed = dataset.train_text_embed
	train_img_embed = dataset.train_img_embed

	test_text_embed = dataset.test_text_embed
	test_text_embed = dataset.test_img_embed
	N = train_text_embed.shape[0]
	dim1 = train_text_embed.shape[-1]
	dim2 = train_img_embed.shape[-1]

	# Hyper parameters
	k = config.k
	bs = config.batch
	act = config.act
	dropout = config.dropout
	output_dim = config.output_dim
	batchs = np.arange(bs, N, bs)
	n_iter = 30

	mlp1 = FCNet([dim1, output_dim * k], act=act, dropout=dropout).to(device)
	mlp2 = FCNet([dim2, output_dim * k], act=act, dropout=dropout).to(device)

	opt = optim.Adam(list(mlp1.parameters()) + list(mlp2.parameters()), lr=0.001, weight_decay=0.01)

	mlp1.train()   
	mlp2.train()   
	loss_lst = []
	for i in range(1, n_iter):
		old_b = 0
		for batch in batchs:
			opt.zero_grad()
			loss = torch.zeros(1).to(device)
			batch_x = text_embed[old_b : batch]
			batch_y = img_embed[old_b : batch]
			x = mlp1(batch_x).reshape(-1, output_dim * (k), 1)
			y = mlp2(batch_y).reshape(-1, output_dim * (k), 1)

			cos_distance = cost_matrix_batch_torch(x.transpose(2, 1), y.transpose(2, 1))
			cos_distance = cos_distance.transpose(1,2)

			beta = 0.1
			min_score = cos_distance.min()
			max_score = cos_distance.max()
			threshold = min_score + beta * (max_score - min_score)
			cos_dist = torch.nn.functional.relu(cos_distance - threshold)

			wd = - IPOT_distance_torch_batch_uniform(C=cos_dist, bs=x.size(0), n=x.size(1), m=y.size(1), iteration=30)
			gwd = GW_distance_uniform(x.transpose(2,1), y.transpose(2,1))

			twd = .1 * torch.mean(gwd) + .1 * torch.mean(wd)
			loss = loss + twd
			# loss = loss - torch.bmm(x.transpose(2,1), y).sum() * 0.01
			# loss = loss + torch.norm(x - y) * 0.1
			old_b = batch
			loss.backward()
			opt.step()
		loss_lst.append(loss.item())
		print(f"Iteration {i} || alignment loss: {loss.item():.4f}")

	mlp1.eval()   
	mlp2.eval()  
	aligned_text_embed = mlp1(text_embed).detach().cpu().numpy()
	aligned_img_embed = mlp2(img_embed).detach().cpu().numpy()
	concat_aligned = np.hstack([aligned_text_embed, aligned_img_embed])

	x_train = concat_aligned[dataset.train_indices]
	x_test = concat_aligned[dataset.valid_indices]
	y_train = dataset.y_train
	y_test = dataset.y_test
	results = sklearn_logress(x_train, x_test, y_train, y_test)
	results['tr_loss'] = loss_lst
	return results
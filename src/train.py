import pdb
import torch
import torch.optim as optim
from clf import *
from fc import *
from ot import *

def train(dataset, config):

	device = config.device
	# text_embed = dataset.text_embed
	# img_embed = dataset.img_embed
	train_text_embed = dataset.train_text_embed
	train_img_embed = dataset.train_img_embed
	test_text_embed = dataset.test_text_embed
	test_img_embed = dataset.test_img_embed

	N = train_text_embed.shape[0]
	dim1 = train_text_embed.shape[-1]
	dim2 = train_img_embed.shape[-1]

	train_indices = dataset.train_indices
	test_indices = dataset.test_indices
	y_train = dataset.y_train
	y_test = dataset.y_test

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
	classifier = FCNet([output_dim * k * 2, 1], act='Sigmoid', dropout=0).to(device)

	opt = optim.Adam(list(mlp1.parameters()) + list(mlp2.parameters()) + list(classifier.parameters()), lr=0.001, weight_decay=0.01)
	criterion = torch.nn.BCELoss()

	loss_lst = []
	for i in range(1, n_iter):
		old_b = 0
		mlp1.train()   
		mlp2.train()  
		classifier.train()
		for batch in batchs:
			opt.zero_grad()
			# loss = torch.zeros(1).to(device)
			batch_x = train_text_embed[old_b : batch]
			batch_y = train_img_embed[old_b : batch]
			x = mlp1(batch_x).reshape(-1, output_dim * (k), 1)
			y = mlp2(batch_y).reshape(-1, output_dim * (k), 1)


			# Classification
			concat_aligned = torch.hstack([x, y])
			batch_y_train = y_train[old_b:batch]
			# pdb.set_trace()
			pred = classifier(concat_aligned.reshape(bs, -1))
			loss = criterion(pred, batch_y_train)
			# pdb.set_trace()
			train_result = pt_evaluate(pred, batch_y_train, verbose=False)

			# Alignment
			cos_distance = cost_matrix_batch_torch(x.transpose(2, 1), y.transpose(2, 1))
			cos_distance = cos_distance.transpose(1,2)

			beta = 0.1
			min_score = cos_distance.min()
			max_score = cos_distance.max()
			threshold = min_score + beta * (max_score - min_score)
			cos_dist = torch.nn.functional.relu(cos_distance - threshold)

			wd = - IPOT_distance_torch_batch_uniform(C=cos_dist, bs=x.size(0), n=x.size(1), m=y.size(1), iteration=30)
			gwd = GW_distance_uniform(x.transpose(2,1), y.transpose(2,1))
			twd = .5 * torch.mean(gwd) + .5 * torch.mean(wd)
			loss = loss + twd


			old_b = batch
			loss.backward()
			opt.step()

		loss_lst.append(loss.item())

		mlp1.eval()   
		mlp2.eval()  
		classifier.eval()
		aligned_test_text_embed = mlp1(test_text_embed)
		aligned_test_img_embed = mlp2(test_img_embed)
		test_concat_aligned = torch.hstack([aligned_test_text_embed, aligned_test_img_embed])
		test_pred = classifier(test_concat_aligned)
		test_result = pt_evaluate(test_pred, y_test, verbose=False)
		print(f"Iteration {i} || total loss: {loss.item():.4f} alignmet loss: {twd.item():.4f} Accuracy: {train_result['acc']:.4f} Test Accuracy: {test_result['acc']:.4f}")

	# results = sklearn_logress(x_train, x_test, y_train, y_test)
	test_result['tr_loss'] = loss_lst
	return test_result
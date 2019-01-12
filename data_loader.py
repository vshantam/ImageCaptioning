#Importing Libraries

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):

	@classmethod
	def __init__(self, root, json, vocab, transform=None):

		self.root = root
		self.coco = COCO(json)
		self.ids = list(self.coco.anns.keys())
		self.vocab = vocab
		self.transform = transform

	@classmethod
	def __getitem__(self, index):

		coco = self.coco
		vocab = self.vocab
		ann_id = self.ids[index]
		caption = coco.anns[ann_id]['caption']
		img_id = coco.anns[ann_id]['image_id']
		path = coco.loadImgs(img_id)[0]['file_name']

		image = Image.open(os.path.join(self.root, path)).convert('RGB')
		if self.transform is not None:
			image = self.transform(image)

		# Convert caption (string) to word ids.
		tokens = nltk.tokenize.word_tokenize(str(caption).lower())
		caption = []
		caption.append(vocab('<start>'))
		caption.extend([vocab(token) for token in tokens])
		caption.append(vocab('<end>'))
		target = torch.Tensor(caption)

		return image, target

	@classmethod
	def __len__(self):
		return len(self.ids)


def collate_fn(data):

	# Sort a data list by caption length (descending order).
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, captions = zip(*data)

	# Merge images (from tuple of 3D tensor to 4D tensor).
	images = torch.stack(images, 0)

	# Merge captions (from tuple of 1D tensor to 2D tensor).
	lengths = [len(cap) for cap in captions]
	targets = torch.zeros(len(captions), max(lengths)).long()
	for i, cap in enumerate(captions):
		end = lengths[i]
		targets[i, :end] = cap[:end]
        
	return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):


	# COCO caption dataset
	coco = CocoDataset(root=root, json=json, vocab=vocab, transform=transform)
    
	# Data loader for COCO dataset

	data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

	return data_loader



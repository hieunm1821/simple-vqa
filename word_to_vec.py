import torch
import numpy as np
def build_vocab():
	return {'yellow': 0, 'no': 1, 'black': 2, 'not': 3, 'does': 4,
			'shape': 5, 'of': 6, 'image': 7, 'gray': 8, 'is': 9, 
			'a': 10, 'what': 11, 'in': 12, 'green': 13, 'color': 14, 
			'circle': 15, 'red': 16, 'the': 17, 'teal': 18, 'rectangle': 19, 
			'brown': 20, 'contain': 21, 'triangle': 22, 'there': 23, 'blue': 24, 'present': 25}

vocab = build_vocab()

def tokenizer(question, vocab):
	vec = np.zeros((1,26))
	question = question.replace("?", "")
	question = question.split(" ")
	for w in question:
		vec[:, vocab[w]] = 1
	return vec

def texts_to_matrix(list_qn, vocab):
	result = []
	for qn in list_qn:
		result.append(tokenizer(qn, vocab))
	return torch.Tensor(np.matrix(np.vstack(result)))
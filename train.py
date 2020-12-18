from easy_vqa import get_train_image_paths, get_test_image_paths, get_train_questions, get_test_questions, get_answers
import torch
from Net import *
from word_to_vec import *
import cv2
import numpy as np
import torch.optim as optim

all_answers = get_answers()
num_ans = len(all_answers)

train_qs, train_answers, train_image_ids = get_train_questions()
test_qs, test_answers, test_image_ids = get_test_questions()

def load_and_proccess_image(image_path):
	im = cv2.imread(image_path)
	return torch.Tensor(im / 255 - 0.5)

def read_images(paths):
	ims = {}
	for image_id, image_path in paths.items():
		ims[image_id] = load_and_proccess_image(image_path)
	return ims

train_ims = read_images(get_train_image_paths())
test_ims = read_images(get_test_image_paths())

train_X_ims = torch.Tensor([train_ims[id] for id in train_image_ids])
test_X_ims = torch.Tensor([test_ims[id] for id in test_image_ids])

vocab = build_vocab()

train_X_seqs = texts_to_matrix(train_qs, vocab)
test_X_seqs  = texts_to_matrix(test_qs, vocab)

def to_categorical(y, num_ans):
    """ 1-hot encodes a tensor """
    return torch.Tensor(np.eye(num_ans, dtype='uint8')[y])

train_answer_indices = [all_answers.index(a) for a in train_answers]
test_answer_indices = [all_answers.index(a) for a in test_answers]
train_Y = to_categorical(train_answer_indices, num_ans)
test_Y = to_categorical(test_answer_indices, num_ans)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
print("Start training")

net = MergeNet(len(vocab), num_ans)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
	running_loss = 0.0
	for i in range(len(train_X_ims)):
		print(i)
		image, question, answer = train_X_ims[i].to(device), train_X_seqs[i].to(device), train_Y[i].to(device)

		optimizer.zero_grad()

		output = net(image, question)
		loss = criterion(output, answer)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0
print("Finished training")
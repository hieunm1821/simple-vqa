import torch
from Net import *
from word_to_vec import *
import cv2
import numpy as np
import torch.optim as optim
import json
import os

with open('data/answers.txt', 'r') as file:
    all_answers = [a.strip() for a in file]
num_ans = len(all_answers)
print('Found {} total answers'.format(num_ans))

def read_questions(path):
	with open(path, 'r') as file:
		qs = json.load(file)
	texts = [q[0] for q in qs]
	answers = [q[1] for q in qs]
	image_ids = [q[2] for q in qs]
	return (texts, answers, image_ids)

train_qs, train_answers, train_image_ids = read_questions('data/train/questions.json')
test_qs, test_answers, test_image_ids = read_questions('data/test/questions.json')

print('Read {} training questions and {} testing questions.'.format(len(train_qs), len(test_qs)))


def extract_paths(dir):
	paths = {}
	for filename in os.listdir(dir):
		if filename.endswith('.png'):
			image_id = int(filename[:-4])
			paths[image_id] = os.path.join(dir, filename)
	return paths

def load_and_proccess_image(image_path):
    im = cv2.imread(image_path)
    im = torch.from_numpy(im)
    im = im.permute(2, 0, 1)
    return (im / 255 - 0.5)

def read_images(paths):
	ims = {}
	for image_id, image_path in paths.items():
		ims[image_id] = load_and_proccess_image(image_path)
	return ims

train_ims = read_images(extract_paths('data/train/images'))
test_ims  = read_images(extract_paths('data/test/images'))

print('Read {} training images and {} testing images.'.format(len(train_ims), len(test_ims)))

train_X_ims = torch.stack([train_ims[id] for id in train_image_ids])
test_X_ims = torch.stack([test_ims[id] for id in test_image_ids])

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

trainset = []
testset = []

for i in range(train_X_ims.shape[0]):
    trainset.append([train_X_ims[i], train_X_seqs[i], train_Y[i]])

for i in range(test_X_ims.shape[0]):
    testset.append([test_X_ims[i], test_X_seqs[i], test_Y[i]])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

trainiter = iter(trainloader)
testiter = iter(testloader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = MergeNet(len(vocab), num_ans)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

print("Start training")

dataset = {"train": trainloader, "val": testloader}
data_lengths = {"train": len(trainset), "val": len(testset)}

for epoch in range(8):
	print("Epoch {}". format(epoch + 1))
	for phase in ["train", "val"]:
		if phase == "train":
			net.train(True)
		else:
			net.train(False)
		running_loss = 0.0
		running_accuracy = 0
		for i, data in enumerate(dataset[phase], 0):
			image, question, answer = data

			image = image.to(device)
			question = question.to(device)
			answer = answer.to(device)
			optimizer.zero_grad()

			output = net(image, question)
			loss = criterion(output, torch.max(answer, 1)[1])
			accuracy = torch.sum(torch.max(answer, 1)[1] == torch.max(output, 1)[1])
			if phase == "train":
				loss.backward()
				optimizer.step()

			running_loss += loss.item()
			running_accuracy += accuracy.item()

		epoch_loss = running_loss / data_lengths[phase]
		epoch_accuracy = running_accuracy / data_lengths[phase]
		
		print('{} loss: {:.4f}'.format(phase, epoch_loss))
		print('{} accuracy: {:.4f}'.format(phase, epoch_accuracy))
print("Finished training")

PATH = './model.pth'
torch.save(net.state_dict(), PATH)
print("Saved model successfully, path: {}".format(PATH))
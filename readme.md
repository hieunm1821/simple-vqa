# Simple Visual Question Answering 

This repository contains Neural network and Demo deploying.

To practice building End-to-end AI application. 

## Data set: [easy-VQA](https://github.com/vzhou842/easy-VQA)

## ConvNet Architecture:

	  conv		maxpool			conv 		maxpool		  flatten	  fc
64x64 ---> 64x64x8 ---> 32x32x8 ---> 32x32x16 ---> 16x16x16 ---> 4096 ---> 32

## Requirements:
- python 3.7
- torch 1.7
- numpy
- easy_vqa


## TODO:
- Train model
- Build demo
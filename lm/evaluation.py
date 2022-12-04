from nltk.translate.bleu_score import sentence_bleu
import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import editdistance

def print_summ(arr):
	print('avg', arr.mean())
	print('std', arr.std())
	print('min', arr.min())
	print('max', arr.max())

def evaluate(x, y, yhat):
	bleus = [sentence_bleu([yi.split()], yhati.split()) for yi, yhati in zip(y, yhat)]
	bleus = np.array(bleus)
	print('BLEU')
	print_summ(bleus)
	
	coverages = []
	order_scores = []
	for ingrs, generated in zip(x, yhat):
		order = [generated.find(ingr) for ingr in ingrs]
		order = [o for o in order if o != -1]
		coverages.append(len(order) / len(ingrs))
		order = rankdata(order, method='ordinal')
		order = ''.join([chr(ord('a') + i) for i in order])
		expected = ''.join([chr(ord('a') + i) for i in range(len(ingrs))])
		dist = editdistance.eval(order, expected)
		order_scores.append(dist)
	coverages = np.array(coverages)
	order_scores = np.array(order_scores)
	print('Coverage')
	print_summ(coverages)
	print('Order score')
	print_summ(order_scores)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, required=True)
	parser.add_argument('--generated_text', type=str, required=True)
	args = parser.parse_args()
	
	y_fn = '../dataset/lm/%s.txt' % args.split
	yhat_fn = args.generated_text
	with open(y_fn) as f:
		x, y = zip(*[l.split(' = ') for l in f.read().split('\n') if l])
	with open(yhat_fn) as f:
		yhat = [l for l in f.read().split('\n') if l]
	assert len(y) == len(yhat), "data set size does not match"
	x = [i.split('<INGR_START>')[1].split('<INGR_END>')[0].split('<NEXT_INGR>') for i in x]
	x = [[ingr.strip() for ingr in line if ingr.strip()] for line in x]
	evaluate(x, y, yhat)

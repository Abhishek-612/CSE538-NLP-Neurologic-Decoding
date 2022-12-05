from nltk.translate.bleu_score import sentence_bleu
import json
import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import editdistance
from rouge import Rouge

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

	rouge = Rouge()
	rouges = [rouge.get_scores(yhati, yi)[0]['rouge-l']['r'] for yi, yhati in zip(y, yhat)]
	rouges = np.array(rouges)
	print('ROUGE-L')
	print_summ(rouges)
	
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
		score = (len(ingrs) - dist) / len(ingrs)
		order_scores.append(score)
	coverages = np.array(coverages)
	order_scores = np.array(order_scores)
	print('Coverage')
	print_summ(coverages)
	print('Order score')
	print_summ(order_scores)
	return bleus.mean(), coverages.mean(), order_scores.mean(), rouges.mean()

def parse_files(y_fn, yconst_fn, yhat_fn):
	with open(y_fn) as f:
		y = [l.split(' = ')[1].replace('. ', ' <NEXT_INSTR> ').lower() for l in f.read().split('\n') if l]
		#y = [l.split(' = ')[1].lower() for l in f.read().split('\n') if l]
	with open(yhat_fn) as f:
		yhat = [l.replace('.', '').lower() for l in f.read().split('\n') if l]
		#yhat = [l.lower() for l in f.read().split('\n') if l]
	x = []
	with open(yconst_fn) as f:
		for l in f.read().split('\n'):
			if not l.strip():
				continue
			ingr = json.loads(l)
			ingr = [i[0][1] if isinstance(i[0][1], str) else i[0][0] for i in ingr]
			x.append(ingr)
	print(len(y), len(yhat))
	assert len(y) == len(yhat), "data set size does not match"
	return x, y, yhat

def hypers(prunes, betas):
	beam = 6
	results = np.zeros((len(prunes),len(betas),4))
	for pi, prune in enumerate(prunes):
	#for pi, prune in enumerate([50, 100, 200]):
		#for bi, beta in enumerate(['1', '5', '10', '50']):
		for bi, beta in enumerate(betas):
			yhat_fn = '%s-%d-%d-%s' % (args.generated_text, beam, prune, beta)
			x, y, yhat = parse_files(y_fn, yconst_fn, yhat_fn)
			print('prune, beta:', prune, beta)
			results[pi,bi,:] = evaluate(x, y, yhat)
	return results
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, required=True)
	parser.add_argument('--generated_text', type=str, required=True)
	parser.add_argument('--line_graph', action='store_true')
	args = parser.parse_args()

	y_fn = '../dataset/lm/%s.txt' % args.split
	yconst_fn = '../dataset/clean/constraint/%s.constraint.json' % args.split
	yhat_fn = args.generated_text
	
	if args.line_graph:
		#prunes = [50, 500, 5000, 50000]
		prunes = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]
		#betas = ['0.001', '0.01', '0.1', '1']
		betas = ['1']
		results = hypers(prunes, betas)
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ln1 = ax1.plot(prunes, results[:,0,0], label='BLEU')
		ln2 = ax2.plot(prunes, results[:,0,1], label='Coverage', color='orange')
		ax1.set_ylabel('BLEU')
		ax2.set_ylabel('Coverage', rotation=270, labelpad=15)
		plt.semilogx()
		labels = [50, 100, 200, 500, 1000, 5000, 10000, 50000]
		ticks = [2 ** i for i in range(5, 17)]
		labels = ['$2^{%d}$' % i for i in range(5, 17)]
		plt.xticks(ticks=ticks, labels=labels)
		plt.minorticks_off()
		ax1.set_xlabel('Pruning Factor')
		plt.title('Effect of Pruning Factor Hyperparameter')
		lns = ln1 + ln2
		labs = [l.get_label() for l in lns]
		plt.legend(lns, labs, loc='center right')
		print(results[:,0,0])
		print(results[:,0,1])
		print(results[:,0,2])
		plt.savefig('plot.pdf')
	else:
		x, y, yhat = parse_files(y_fn, yconst_fn, yhat_fn)
		evaluate(x, y, yhat)


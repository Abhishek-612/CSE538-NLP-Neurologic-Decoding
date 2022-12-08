# convert data samples in individual files into format read in by decode.sh from neurologic decoding
# also create json constraints file
import sys
import json
import os
from os.path import join
import argparse

def reformat(x):
	x = '<RECIPE_START> ' + x
	x = x.replace('<start-title>', '<TITLE_START>')
	x = x.replace('<end-title>', '<TITLE_END>')
	x = x.replace('<start-ingredients>', '<INGR_START> ')
	x = x.replace('<end-ingredients>', '<INGR_END>')
	x = x.replace('$', ' <NEXT_INGR> ')
	x = x.replace('<start-directions>', '<INSTR_START>')
	return x

def parse_ingredients(x):
	ingr = x.split('<INGR_START>')[1].split('<INGR_END>')[0]
	ingr = [i.strip() for i in ingr.split('<NEXT_INGR>')]
	ingr = [i for i in ingr if i]
	return ingr

def positive_ingredients(i):
	ret = [[i, True]]
	words = i.split(' ')
	if len(words) > 1:
		pass
		#ret.extend([[w, True] for w in words])
	return ret

def until_ingredients(ingr):
	const = []
	for i in range(len(ingr)-1):
		# not b until a
		const.append([[ingr[i+1], ingr[i]]])
	const.append([[ingr[-1], True]])
	return const

def convert(output_fn, constraint_type, small_n):
	samples = os.listdir('X')
	outf = open(output_fn + '.txt', 'w')
	conf = open(output_fn + '.constraint.json', 'w')
	i = 0
	for sample in samples:
		if sample[-5] != 'd':
			continue
		with open(join('X', sample)) as f:
			x = f.read().strip()
		with open(join('y', sample)) as f:
			y = f.read().strip()
		x = reformat(x)
		outf.write('%s = %s\n' % (x, y))
		ingr = parse_ingredients(x)
		if constraint_type == 'until':
			const = until_ingredients(ingr)
		elif constraint_type == 'positive':
			const = [positive_ingredients(ing) for ing in ingr]
		elif constraint_type == 'none':
			const = []
		conf.write('%s\n' % json.dumps(const))
		i += 1
		if i == small_n:
			break
	outf.close()
	conf.close()
	print('wrote %s samples to %s' % (small_n, output_fn))

parser = argparse.ArgumentParser()
parser.add_argument('--output_fn', required=True)
parser.add_argument('--small_n', default=0, type=int)
parser.add_argument('--constraint_type', type=str, default='positive')
args = parser.parse_args()
convert(**vars(args))
#convert(args.output_fn, args.until, args.small_n)

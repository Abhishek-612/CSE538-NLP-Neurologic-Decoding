from itertools import permutations
import editdistance
from collections import Counter
import json

def get_dist(n):
	inorder = ''.join([chr(ord('a') + i) for i in range(n)])
	dists = []
	for l in range(n + 1):
		i = 0
		for p in permutations(inorder, l):
			i += 1
			dist = editdistance.distance(''.join(p), inorder)
			dists.append(dist)
	counts = Counter(dists)
	return counts


if __name__ == '__main__':
	data = []
	for n in range(11):
		data.append(get_dist(n))
	with open('editdistance_distributions.json', 'w') as f:
		json.dump(data, f)

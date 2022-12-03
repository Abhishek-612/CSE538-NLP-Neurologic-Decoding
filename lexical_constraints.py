import copy
import logging
from operator import attrgetter
from typing import Dict, List, Optional, Tuple, Set, Union
import collections

import numpy as np
import torch

logger = logging.getLogger(__name__)

Phrase = List[int]
Literal = Tuple[Phrase, bool]
# Represents a list of raw constraints for a sentence. Each constraint is a list of target-word IDs.
RawConstraintList = List[Phrase]
ClauseConstraintList = List[List[Literal]]


class Trie:
	"""
	Represents a set of phrasal constraints for an input sentence.
	These are organized into a trie.
	"""
	def __init__(self, clause_idx: int = None,
				 raw_phrases: Optional[RawConstraintList] = None,
				 parent_arc: int = None,
				 parent_trie: 'Trie' = None) -> None:
		self.final_ids = {}	# type: Dict[int, Set[int]] (token_id -> set(clause_idx))
		self.children = {}	# type: Dict[int,'Trie']
		self.parent_arc = parent_arc
		self.parent_trie = parent_trie

		if raw_phrases:
			for phrase in raw_phrases:
				self.add_phrase(phrase, clause_idx)

	def add_phrase(self,
				   phrase: List[int], clause_idx: int) -> None:
		"""
		Recursively adds a phrase to this trie node.

		:param phrase: A list of word IDs to add to this trie node.
		"""
		if len(phrase) == 1:
			self.final_ids.setdefault(phrase[0], set()).add(clause_idx)
		else:
			next_word = phrase[0]
			if next_word not in self.children:
				self.children[next_word] = Trie(clause_idx=clause_idx, parent_arc=next_word, parent_trie=self)
			self.step(next_word).add_phrase(phrase[1:], clause_idx)

	def delete_phrase(self,
					  phrase: List[int], clause_idx: int) -> None:
		"""
		Recursively deletes a phrase to this trie node.

		:param phrase: A list of word IDs to delete in this trie node.
		"""
		if len(phrase) == 1:
			assert phrase[0] in self.final_ids, f"Trie {str(self)} \nDo not contain {phrase}"
			assert clause_idx in self.final_ids[phrase[0]], f"Trie {str(self)} does not contain clause_idx {clause_idx}"
			self.final_ids[phrase[0]].remove(clause_idx)
			if not len(self.final_ids[phrase[0]]):
				del self.final_ids[phrase[0]]
		else:
			next_word = phrase[0]
			assert next_word in self.children.keys(), f"Trie {str(self)} \nDo not contain {phrase}"
			self.step(next_word).delete_phrase(phrase[1:], clause_idx)

			# Move the arc to an empty node to final_ids of its parent
			for arc in list(self.children):
				if len(self.children[arc]) == 0:
					self.children.pop(arc)

	def check_phrase(self,
					 phrase: List[int]) -> bool:
		"""
		Check whether a phrase is in this trie.

		:param phrase: A list of word IDs to check existence.
		"""
		if len(phrase) == 1:
			return phrase[0] in self.final_ids
		else:
			next_word = phrase[0]
			if next_word in self.children:
				return self.step(next_word).check_phrase(phrase[1:])
			return False

	def trace_phrase(self,
					 word_id: int) -> List[int]:
		"""
		Recursively backward to get word ids in a phrase.

		:param word_id: The last word IDs in phrase.
		"""
		assert word_id in self.final_ids, f"{word_id} does not in trie node {self.final_ids}"
		phrase = self.trace_arcs()
		phrase.append(word_id)
		return phrase

	# ?? this isn't recursive... not that it matters...
	def trace_arcs(self,) -> List[int]:
		"""
		Recursively backward to get arc to ancestor
		"""
		arcs = []
		parent_trie, parent_arc = self.parent_trie, self.parent_arc
		while parent_trie is not None:
			arcs.append(parent_arc)
			parent_arc = parent_trie.parent_arc
			parent_trie = parent_trie.parent_trie
		arcs.reverse()
		return arcs

	def __str__(self) -> str:
		s = f'({list(self.final_ids)}'
		for child_id in self.children.keys():
			s += f' -> {child_id} {self.children[child_id]}'
		s += ')'
		return s

	def __len__(self) -> int:
		"""
		Returns the number of phrases represented in the trie.
		"""
		phrase_count = len(self.final_ids)
		for child in self.children.values():
			phrase_count += len(child)
		return phrase_count

	def step(self, word_id: int) -> Optional['Trie']:
		"""
		Returns the child node along the requested arc.

		:param word_id: requested arc.
		:return: The child node along the requested arc, or None if no such arc exists.
		"""
		return self.children.get(word_id, None)

	def descend(self,
			  arcs: List[int]) -> Optional['Trie']:
		pointer = self
		for arc in arcs:
			if pointer is None:
				break
			pointer = pointer.step(word_id=arc)
		return pointer

	def final(self) -> Set[int]:
		"""
		Returns the set of final ids at this node.

		:return: The set of word IDs that end a constraint at this state.
		"""
		return set(self.final_ids.keys())

# this shouldn't even be a class
# the trie should have pi links to efficiently find phrases
# instead they naively make a new trie for every time step
# and use this class to manage their set of tries
class TrieManager:
	"""
	Represents a set of words and phrases that must appear in the output.
	The offset is used to return actual positions in the one-dimensionally-resized array that
	get set to infinity.

	:param positive_trie: The trie containing the phrases to appear.
	:param state: The current state (defaults to root).
	"""
	def __init__(self,
				 positive_trie: Trie,
				 state: List[Trie] = None,
				 met_phrases: RawConstraintList = None) -> None:

		self.root = positive_trie
		self.state = state if state else [self.root]
		self.met_phrases = met_phrases if met_phrases else set()

	def __str__(self):
		s = f'Root: {self.root}\nState: ['
		for state in self.state:
			s += f'{state}, '
		s += f']\nMet_phrases: {self.met_phrases}'
		return s
	
	def prune_states(self):
		new_states = set()
		for s in self.state:
			if s.parent_trie is None:
				new_states.add(s)
			else:
				trace = s.trace_arcs()
				new_state = self.root.descend(trace)
				if new_state:
					new_states.add(new_state)
		self.state = list(new_states)

	def allowed(self) -> Set[int]:
		"""
		Returns the set of constrained words that could follow this one.
		For unfinished phrasal constraints, it is the next word in the phrase.
		In other cases, it is the list of all unmet constraints.
		If all constraints are met, an empty set is returned.

		:return: The ID of the next required word, or -1 if any word can follow
		"""
		allow = self.root.final().union(*[state.final() for state in self.state])
		allow |= set(self.root.children.keys()).union(*[set(state.children.keys()) for state in self.state])
		return allow

	def advance(self, word_id: int) -> 'TrieManager':
		"""
		Updates the constraints object based on advancing on word_id.
		There is a complication, in that we may have started but not
		yet completed a multi-word constraint.	We need to allow constraints
		to be added as unconstrained words, so if the next word is
		invalid, we must "back out" of the current (incomplete) phrase,
		re-setting all of its words as unmet.

		:param word_id: The word ID to advance on.
		:return: A deep copy of the object, advanced on word_id.
		"""
		new_state, met_phrases = [], set()
		for state in set(self.state + [self.root]):
			if word_id in state.children:
				new_state.append(state.step(word_id))
			if word_id in state.final_ids:
				met_phrases.add(tuple(state.trace_phrase(word_id)))

		if new_state:
			return TrieManager(self.root, new_state, met_phrases)
		else:
			# why do we check if met_phrases is empty? why not self.met_phrases = met_phrases?
			if len(self.state) == 1 and self.root == self.state[0] and not met_phrases:
				return self
			else:
				return TrieManager(self.root, [self.root], met_phrases)


class Clause:
	"""
	Object used to hold clause.

	:param idx: The id of this clause.
	:param positive: The positive constraints in this clause.
	:param negative: The soft negative constraints in this clause.
	:param satisfy: whether this clause is satisfied
	"""

	__slots__ = ('idx', 'positive', 'negative', 'until', 'satisfy', 'reversible')

	def __init__(self,
				 idx: int,
				 positive: List[Phrase],
				 negative: List[Phrase],
				 until) -> None:
		self.idx = idx
		self.positive = {tuple(p) for p in positive}
		self.negative = {tuple(n) for n in negative}
		self.until = dict()
		for a, b in until:
			self.until.setdefault(a, set()).add(b)
		self.reversible = True
		self.set_satisfy()

	def __str__(self):
		return f'clause(id={self.idx}, positive={self.positive}, negative={self.negative}, satisfy={self.satisfy})'
	
	def set_satisfy(self):
		self.satisfy = bool(len(self.negative - set(self.until.keys())))
	
	def met_phrase(self, phrase):
		assert self.reversible == True, "met_phrase called on irreversible clause"
		if phrase in self.positive:
			self.satisfy = True
			self.reversible = False
			return self.negative, self.positive
		if phrase in self.negative:
			self.negative.remove(phrase)
			pos_phrases = self.until.pop(phrase, set())
			for pos_phrase in pos_phrases:
				self.positive.remove(pos_phrase)
			self.set_satisfy()
			self.reversible = bool(len(self.positive)) or self.satisfy
			return set([phrase]), pos_phrases
		return set(), set()

def is_prefix(pref: List[int],
			  phrase: List[int]):
	if not pref:
		return False
	return pref == phrase[:len(pref)]


class ConstrainedHypothesis:
	"""
	Keep track of positive and negative constraint

	hard negative constraint will not be generated in any cases
	soft negative constraint might be generated in some case due to OR gate in clause
	positive constraints will be encourage to appear

	:param constraint_list: A list of clause constraints (each represented as a list of literals).
	"""
	def __init__(self,
				 constraint_list: ClauseConstraintList,
				 eos_id: Union[int, list]
				 ) -> None:
		self.eos_id = eos_id if isinstance(eos_id, list) else [eos_id]
		self.clauses = []  # type: List[Clause]

		t_pos = Trie()
		t_neg = Trie()
		for idx, clause in enumerate(constraint_list):
			if not clause:
				continue
			pos_phrases, neg_phrases, untils = [], [], []
			for l in clause:
				a, b = l
				if isinstance(b, tuple):
					untils.append((a, b))
					neg_phrases.append(a)
					pos_phrases.append(b)
				elif b == True:
					pos_phrases.append(a)
				elif b == False:
					neg_phrases.append(a)
				else:
					assert False, 'constraint list malformed type'
			for p in neg_phrases:
				t_neg.add_phrase(p, idx)
			for p in pos_phrases:
				t_pos.add_phrase(p, idx)
			self.clauses.append(Clause(idx=idx, positive=pos_phrases, negative=neg_phrases, until=untils))

		self.negative_state = TrieManager(t_neg)
		self.positive_state = TrieManager(t_pos)

		self.orders = []
		self.in_process = None
		self.max_process = 0
		self.valid = True

	def __len__(self) -> int:
		"""
		:return: The number of constraints.
		"""
		return len(self.clauses)

	def __str__(self) -> str:
		return '\n'.join([str(c) for c in self.clauses])

	def size(self) -> int:
		"""
		:return: the number of constraints
		"""
		return len(self.clauses)

	def num_met(self) -> int:
		"""
		:return: the number of constraints that have been met.
		"""
		if not self.clauses:
			return 0
		return sum([int(c.satisfy) for c in self.clauses])

	def met_order(self) -> tuple:
		"""
		:return: the constraints that have irreversibly been met.
		"""
		return tuple(sorted(self.orders))

	def clause_in_process(self) -> tuple:
		"""
		:return: the index of clause that's in generation.
		"""
		return tuple(self.in_process)

	def num_needed(self) -> int:
		"""
		:return: the number of un-met constraints.
		"""
		return self.size() - self.num_met()

	def finished(self) -> bool:
		"""
		Return true if all the constraints have been met.

		:return: True if all the constraints are met.
		"""
		return self.num_needed() == 0

	def is_valid(self, wordid: int) -> bool:
		"""
		Ensures </s> is only generated when the hypothesis is completed.

		:param wordid: The wordid to validate.
		:return: True if all constraints are already met or the word ID is not the EOS id.
		"""
		return self.finished() or self.valid and wordid not in self.eos_id

	def eos(self) -> list:
		"""
		:return: Return EOS id.
		"""
		return self.eos_id
	
	def allowed(self):
		return self.positive_state.allowed() - self.negative_state.allowed()

	def advance(self, word_id: int) -> 'ConstrainedHypothesis':
		"""
		Updates the constraints object based on advancing on word_id.
		If one of literals in a clause is satisfied, we mark this clause as satisfied

		:param word_id: The word ID to advance on.
		"""
		obj = copy.deepcopy(self)
		
		obj.negative_state = obj.negative_state.advance(word_id)
		obj.positive_state = obj.positive_state.advance(word_id)
		met_phrases = obj.negative_state.met_phrases | obj.positive_state.met_phrases

		for phrase in met_phrases:
			for clause in obj.clauses:
				if not clause.reversible:
					continue
				del_neg, del_pos = clause.met_phrase(phrase)
				# delete unneeded literals
				for p in del_neg:
					if obj.negative_state.root.check_phrase(p):
						obj.negative_state.root.delete_phrase(p, clause.idx)
				for p in del_pos:
					if obj.positive_state.root.check_phrase(p):
						obj.positive_state.root.delete_phrase(p, clause.idx)
				# update obj.orders
				if not clause.reversible:
					if clause.satisfy:
						obj.orders.append(clause.idx)
					else:
						obj.valid = False

		# have TrieManagers prune invalid states
		obj.negative_state.prune_states()
		obj.positive_state.prune_states()

		# check for in process positive phrases
		history = [s.trace_arcs() for s in obj.positive_state.state]
		newly_in_process = set()
		max_process = 0
		for phrase in history:
			for clause in obj.clauses:
				phrase_in_process = [c for c in clause.positive if is_prefix(phrase, c)]
				if not clause.satisfy and bool(phrase_in_process):
					process_portion = len(phrase) / min([len(x) for x in phrase_in_process])
					max_process = max(max_process, process_portion)
					assert clause.idx not in obj.orders, 'clause has already satisfied, impossible state'
					newly_in_process.add(clause.idx)
		obj.in_process = sorted(newly_in_process)
		obj.max_process = max_process
		return obj


def init_batch(raw_constraints: List[ClauseConstraintList],
			   beam_size: int,
			   eos_id: Union[int, list]) -> List[Optional[ConstrainedHypothesis]]:
	"""
	:param raw_constraints: The list of clause constraints.
	:param beam_size: The beam size.
	:param eos_id: The target-language vocabulary ID of the EOS symbol.
	:return: A list of ConstrainedHypothesis objects (shape: (batch_size * beam_size,)).
	"""
	constraints_list = [None] * (len(raw_constraints) * beam_size)	# type: List[Optional[ConstrainedHypothesis]]
	for i, raw_list in enumerate(raw_constraints):
		hyp = ConstrainedHypothesis(raw_list, eos_id)
		idx = i * beam_size
		constraints_list[idx:idx + beam_size] = [copy.deepcopy(hyp) for _ in range(beam_size)]
	return constraints_list


class ConstrainedCandidate:
	"""
	Object used to hold candidates for the beam in topk().

	:param row: The row in the scores matrix.
	:param col: The column (word ID) in the scores matrix.
	:param score: the associated accumulated score.
	:param hypothesis: The ConstrainedHypothesis containing information about met constraints.
	"""

	__slots__ = ('row', 'col', 'score', 'hypothesis', 'rank')

	def __init__(self,
				 row: int,
				 col: int,
				 score: float,
				 hypothesis: ConstrainedHypothesis,
				 rank: float = None,) -> None:
		self.row = row
		self.col = col
		self.score = score
		self.hypothesis = hypothesis
		self.rank = rank

	def __hash__(self):
		return hash((self.row, self.col))

	def __eq__(self, other):
		return self.row == other.row and self.col == other.col

	def __str__(self):
		return '({}, {}, {}, {})'.format(self.row, self.col, self.score, self.hypothesis.num_met())


if __name__ == '__main__':
	clauses = [[[([3, 4, 5], True), ([3, 4], True), ([4, 5], True)], [([3, 4], True), ([6], True), ([7], True)]],
			   [[([6], True), ([6, 7], True), ([6, 7, 8], True)], [([6, 9], True), ([6, 4, 9], True)]],
			   [[([3, 4, 5], True)], [([3, 4], True)], [([4, 5], True)]],
			   [[([3, 4], True)], [([2, 3, 5], True)], [([6, 5], True)]]]

	constraints = init_batch(raw_constraints=clauses,
							 beam_size=1,
							 eos_id=0)

	constraint = constraints[2]
	print(constraint)
	print(constraints)
	print()
	for w in [2, 3, 4, 5]:
		constraint = constraint.advance(w)
		print(constraint)
		print(constraint.positive_state)
		print(constraint.positive_state.allowed())
		print(constraint.met_order())
		print(constraint.clause_in_process())
		print()



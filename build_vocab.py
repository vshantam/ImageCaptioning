#Importing Libraries
import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


#Defining class for vocabulary.
class Vocabulary(object):

	#Initialising indexes using Dictionary

	def __init__(self):

		#try using default dict( More efficient)
		self.word2idx = {}
		self.idx2word = {}
		self.idx = 0

	#Adding Word to dictionary

	def add_word(self, word):
		
		#If word is not present
		if not word in self.word2idx:
			self.word2idx[word] = self.idx
			self.idx2word[self.idx] = word
			self.idx += 1

	#Re Instantiating Object

	def __call__(self, word):
		if not word in self.word2idx:
			return self.word2idx['<unk>']

		return self.word2idx[word]
	
	#Calculating Length

	def __len__(self):
		return len(self.word2idx)

class Builder(object):


	def build_vocab(self,json, threshold):

		coco = COCO(json)
		counter = Counter()
		ids = coco.anns.keys()
		for i, id in enumerate(ids):
			caption = str(coco.anns[id]['caption'])
			tokens = nltk.tokenize.word_tokenize(caption.lower())
			counter.update(tokens)

			if (i+1) % 1000 == 0:
				print("[{0}/{1}] Tokenized the captions.".format(i+1, len(ids)))

		# If the word frequency is less than 'threshold', then the word is discarded.
		words = [word for word, cnt in counter.items() if cnt >= threshold]

		# Create a vocab wrapper and add some special tokens.
		vocab = Vocabulary()
		vocab.add_word('<pad>')
		vocab.add_word('<start>')
		vocab.add_word('<end>')
		vocab.add_word('<unk>')

		# Add the words to the vocabulary.
		for i, word in enumerate(words):
			vocab.add_word(word)
    
		return vocab


	def main(self,args):
		vocab = self.build_vocab(json=args.caption_path, threshold=args.threshold)
		vocab_path = args.vocab_path
		with open(vocab_path, 'wb') as f:
			pickle.dump(vocab, f)
		print("Total vocabulary size: {}".format(len(vocab)))
		print("Saved the vocabulary wrapper to '{0}'".format(vocab_path))


if __name__ == '__main__':
    
	#Creating Parser objects
	parser = argparse.ArgumentParser()

	parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2017.json', help='path for train annotation file')
	parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for saving vocabulary wrapper')
	parser.add_argument('--threshold', type=int, default=4, help='minimum word count threshold')

	args = parser.parse_args()
	
	#creating BUilder object
	obj = Builder()
	obj.main(args)

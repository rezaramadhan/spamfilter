#!/usr/bin/python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import os, sys, stat, re, string
import shutil
import codecs

srcdir = "extracted"
dstdir = "cleaned"
regex = re.compile('[%s0-9]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

def CleanData(srcpath):
    with codecs.open(srcpath, "r", encoding='ascii', errors='ignore') as file_in:
        # tokenization
        tokenized_data = word_tokenize(file_in.read())
        # print tokenized_data

        # remove punctuation and numbers
        no_punct = []
        for token in tokenized_data:
            token = regex.sub('', token)
            if (token):
                no_punct.append(token)
        # print no_punct

        # clean stopwords
        no_stopwords = []
        for token in no_punct:
            if token not in stopwords.words('english'):
                no_stopwords.append(token)

        # print no_stopwords

        # stemming and lemmatization -> Belum maksimal, masih bisa di optimize
        #                               e.g. POS tagging dulu, terus lemmatize pake wordnet
        
        # porter = PorterStemmer()
        snowball = SnowballStemmer('english')
        # wordnet = WordNetLemmatizer()

        root_tokens = []
        for token in no_stopwords:
            # root_tokens.append(porter.stem(token))
            root_tokens.append(snowball.stem(token))
            # root_tokens.append(wordnet.lemmatize(token))
        # print root_tokens

        return ' '.join(root_tokens)

def FindFileFromDir ( srcdir, dstdir ):
	if not os.path.exists(dstdir): # dest path doesnot exist
		os.makedirs(dstdir)
	files = os.listdir(srcdir)
	for file in files:
		srcpath = os.path.join(srcdir, file)
		dstpath = os.path.join(dstdir, file)
		src_info = os.stat(srcpath)
		if stat.S_ISDIR(src_info.st_mode): # for subfolders, recurse
			ExtractBodyFromDir(srcpath, dstpath)
		else:  # copy the file
			cleaned_data = CleanData (srcpath)
			dstfile = open(dstpath, 'w')
			dstfile.write(cleaned_data)
			dstfile.close()

if __name__ == '__main__':
    FindFileFromDir( srcdir, dstdir )

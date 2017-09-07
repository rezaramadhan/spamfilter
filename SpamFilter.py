#!/usr/bin/python
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

import numpy
import os

LABEL_FILE = "SPAMTrain.label"
DATA_DIR = "cleaned"
data = []

def load_label(labelfile):
    label = {}
    with open(labelfile, "r") as file_in:
        for line in file_in:
            line = line.strip('\n').split(' ')

            label[line[1]] = "HAM" if line[0] == "1" else "SPAM"
    return label

def load_data(datadir, label):
    rows = []
    index = []

    filenames = os.listdir(datadir)
    # print filenames
    for filename in filenames:
        srcpath = os.path.join(datadir, filename)
        index.append(srcpath)
        with open(srcpath, "r") as file_in:
            file_content = file_in.read()

        rows.append({'text': file_content, 'class': label[filename]})

    data_frame = DataFrame(rows, index=index)

    return data_frame

if __name__ == '__main__':
    label = load_label(LABEL_FILE)
    # print label
    data = load_data(DATA_DIR, label)
    count_vectorizer = CountVectorizer()
    counts = count_vectorizer.fit_transform(data['text'].values)

    classifier = MultinomialNB()
    targets = data['class'].values
    classifier.fit(counts, targets)
    examples = ['Free Viagra call today!', "I'm going to attend the Linux users group tomorrow."]
    example_counts = count_vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)
    print predictions # [1, 0]

    pipeline = Pipeline([
        ('vectorizer',  CountVectorizer()),
        ('classifier',  MultinomialNB()) ])

    k_fold = KFold(n=len(data), n_folds=6)
    scores = []
    confusion = numpy.array([[0, 0], [0, 0]])
    for train_indices, test_indices in k_fold:
        train_text = data.iloc[train_indices]['text'].values
        train_y = data.iloc[train_indices]['class'].values

        test_text = data.iloc[test_indices]['text'].values
        test_y = data.iloc[test_indices]['class'].values

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label='SPAM')
        scores.append(score)

    print('Total emails classified:', len(data))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)

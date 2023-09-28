import numpy as np 
import math
from collections import Counter, defaultdict

def handle_unknown_words(t, documents): 
    word_count = {}
    for sentence in documents:
        for word in sentence:
            word_count[word] = word_count.get(word, 0) + 1

    keys = word_count.keys()

    num_to_replace = int(t * len(keys))

    sorted_data = sorted(word_count.items(), key = lambda item: item[1], reverse = False)

    output = []
    for d in documents:
        output.append([])
        for w in d:
            output[-1].append(w)
    for i in range(0, num_to_replace):
        word = sorted_data[i][0]
        for s in range(0,len(output)):
            for w in range(0, len(output[s])):
                if output[s][w] == word:
                    output[s][w] = "<unk>"

    vocab=list({token for sentence in output for token in sentence if token != "<unk>"})
    vocab.append("<unk>")

    return output, vocab
        


def apply_smoothing(k, observation_counts, unique_obs):   
    total = {}
    for (tag, token) in observation_counts.keys():
        observation_counts[(tag, token)] += k
        if tag not in total:
            total[tag] = observation_counts[(tag, token)]
        else:
            total[tag] += observation_counts[(tag, token)]

    d = {}
    for (tag, token) in observation_counts.keys():
        d[(tag, token)] = np.log(observation_counts[(tag, token)] / total[tag])

    return d

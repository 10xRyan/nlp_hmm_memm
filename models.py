# Name(s):Ryan Ren, Victor Fuentes
# Netid(s):yr269, vmf24
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your 
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
from collections import defaultdict
from nltk import classify
from nltk import download
from nltk import pos_tag
import numpy as np

download('averaged_perceptron_tagger')

class HMM: 

  def __init__(self, documents, labels, vocab, all_tags, k_t, k_e, k_s, smoothing_func): 
    """
    Initializes HMM based on the following properties.

    Input:
      documents: List[List[String]], dataset of sentences to train model
      labels: List[List[String]], NER labels corresponding the sentences to train model
      vocab: List[String], dataset vocabulary
      all_tags: List[String], all possible NER tags 
      k_t: Float, add-k parameter to smooth transition probabilities
      k_e: Float, add-k parameter to smooth emission probabilities
      k_s: Float, add-k parameter to smooth starting state probabilities
      smoothing_func: (Dict<key Tuple[String, String] : value Float>, Float) -> 
      Dict<key Tuple[String, String] : value Float>
    """
    self.documents = documents
    self.labels = labels
    self.vocab = vocab
    self.all_tags = all_tags
    self.k_t = k_t
    self.k_e = k_e
    self.k_s = k_s
    self.smoothing_func = smoothing_func
    self.emission_matrix = self.build_emission_matrix()
    self.transition_matrix = self.build_transition_matrix()
    self.start_state_probs = self.get_start_state_probs()


  def build_transition_matrix(self):
    """
    Returns the transition probabilities as a dictionary mapping all possible
    (tag_{i-1}, tag_i) tuple pairs to their corresponding smoothed 
    log probabilities: log[P(tag_i | tag_{i-1})]. 
    
    Note: Consider all possible tags. This consists of everything in 'all_tags', but also 'qf' our end token.
    Use the `smoothing_func` and `k_t` fields to perform smoothing.

    Output: 
      transition_matrix: Dict<key Tuple[String, String] : value Float>
    """
    # YOUR CODE HERE

    # Scan labels and check probability of tag_i given tag_{i-1}

    d = {}
    for tag1 in self.all_tags:
      for tag2 in self.all_tags:
        d[(tag1, tag2)] = 0
      d[(tag1, 'qf')] = 0

    totals = {}
    for label_group in self.labels: # each sentence
      for label in label_group: # each word
        if label not in totals:
          totals[label] = 1
        else:
          totals[label] += 1
      for i in range(1, len(label_group)):
        d[(label_group[i-1], label_group[i])] += 1
      d[(label_group[-1], 'qf')] += 1


    # Divide by number of times tag_{i-1} appears
    for (tag1, tag2) in d:
      d[(tag1, tag2)] = d[(tag1, tag2)] / totals[tag1]

    return self.smoothing_func(d, self.k_t)

  def build_emission_matrix(self): 
    """
    Returns the emission probabilities as a dictionary, mapping all possible 
    (tag, token) tuple pairs to their corresponding smoothed log probabilities: 
    log[P(token | tag)]. 
    
    Note: Consider all possible tokens from the list `vocab` and all tags from 
    the list `all_tags`. Use the `smoothing_func` and `k_e` fields to perform smoothing.
  
    Output:
      emission_matrix: Dict<key Tuple[String, String] : value Float>
      Its size should be len(vocab) * len(all_tags).
    """
    # YOUR CODE HERE
    
    d = {}
    for token in self.vocab:
      for tag in self.all_tags:
        d[(token, tag)] = 0

    totals = {}
    for i in range(0, len(self.documents)):
      document = self.documents[i]
      label_group = self.labels[i]
      for token in document:
        if token not in totals:
          totals[token] = 1
        else:
          totals[token] += 1
      for j in range(0, len(document)):
        d[(document[j], label_group[j])] += 1

    # Divide by number of times tag_{i-1} appears
    for (token, tag) in d:
      d[(token, tag)] = d[(token, tag)] / totals[token]

    return self.smoothing_func(d, self.k_e)
    
  def get_start_state_probs(self):
    """
    Returns the starting state probabilities as a dictionary, mapping all possible 
    tags to their corresponding smoothed log probabilities. Use `k_s` smoothing
    parameter to manually perform smoothing.
    
    Note: Do NOT use the `smoothing_func` function within this method since 
    `smoothing_func` is designed to smooth state-observation counts. Manually
    implement smoothing here.

    Output: 
      start_state_probs: Dict<key String : value Float>
    """
    # YOUR CODE HERE 

    d = {}
    for tag in self.all_tags:
      d[tag] = 0

    for label_group in self.labels:
      tag = label_group[0]
      if tag not in d:
        d[tag] = self.k_s
      else:
        d[tag] += self.k_s

    for tag in d:
      d[tag] = np.log(d[tag] / len(self.labels))

    return d


  def get_trellis_arc(self, predicted_tag, previous_tag, document, i): 
    """
    Returns the trellis arc used by the Viterbi algorithm for the label 
    `predicted_tag` conditioned on the `previous_tag` and `document` at index `i`.
    
    For HMM, this would be the sum of the smoothed log emission probabilities and 
    log transition probabilities: 
    log[P(predicted_tag | previous_tag))] + log[P(document[i] | predicted_tag)].
    
    Note: Treat unseen tokens as an <unk> token.
    Note: Make sure to handle the case where we are dealing with the first word. Is there a transition probability for this case?
    Note: Make sure to handle the case where the predicted tag is an end token. Is there an emission probability for this case?
  
    Input: 
      predicted_tag: String, predicted tag for token at index `i` in `document`
      previous_tag: String, previous tag for token at index `i` - 1
      document: List[String]
      i: Int, index of the `document` to compute probabilities 
    Output: 
      result: Float
    """
    # YOUR CODE HERE

    token = document[i]
    if token not in self.vocab:
      token = "<unk>"

    if i == 0:
      return self.start_state_probs[predicted_tag] + self.emission_matrix[(token, predicted_tag)]
    elif predicted_tag == 'qf':
      return self.transition_matrix[(previous_tag, predicted_tag)]
    else:
      return self.transition_matrix[(previous_tag, predicted_tag)] + self.emission_matrix[(token, predicted_tag)]
 

################################################################################
################################################################################



class MEMM: 

  def __init__(self, documents, labels): 
    """
    Initializes MEMM based on the following properties.

    Input:
      documents: List[List[String]], dataset of sentences to train model
      labels: List[List[String]], NER labels corresponding the sentences to train model
    """
    self.documents = documents
    self.labels = labels
    self.classifier = self.generate_classifier()


  def extract_features_token(self, document, i, previous_tag):
    """
    Returns a feature dictionary for the token at document[i].

    Input: 
      document: List[String], representing the document at hand
      i: Int, representing the index of the token of interest
      previous_tag: string, previous tag for token at index `i` - 1

    Output: 
      features_dict: Dict<key String: value Any>, Dictionaries of features 
                    (e.g: {'Is_CAP':'True', . . .})
    """
    features_dict = {}

    # YOUR CODE HERE 
    
    #First Token
    features_dict['Is_First'] = (1 if i == 0 else 0)

    #Last Toekn
    features_dict['Is_Last'] = (1 if i == len(document)-1 else 0)
    
    #UPPER
    features_dict['Is_Upper'] = (1 if document[i].isupper() else 0)

    #Whole word is lower
    features_dict['Is_Lower'] = (1 if document[i].islower() else 0)

    #Title
    features_dict['Is_Title'] = (1 if document[i].istitle() else 0)

    #Digit Special characters
    character_list = ['(',')',',','[',']',':',';','/','?','{','}','|','@','!','#','$','%','^']
    features_dict['Is_Special'] = 1 if (document[i].isdigit()) or (document[i] in character_list) else 0

    
    features_dict['Token'] = document[i]
    features_dict['Previous_Token'] = document[i-1] if i > 0 else None
    features_dict['Next_Token'] = document[i+1] if i < len(document)-1 else None

    features_dict['Previous_Tag'] = previous_tag
    features_dict['Previous_Is_Upper'] = (1 if document[i-1].isupper() else 0) if i > 0 else None
    features_dict['Previous_Is_Lower'] = (1 if document[i-1].islower() else 0) if i > 0 else None
    features_dict['Previous_Is_Title'] = (1 if document[i-1].istitle() else 0) if i > 0 else None
    features_dict['Previous_Is_Special'] = 1 if (document[i-1].isdigit()) or (document[i-1] in character_list) else 0 if i > 0 else None
    
    features_dict['Previous_Is_B'] = (1 if previous_tag.startswith('B-') else 0) if i > 0 else None
    features_dict['Previous_Is_I'] = (1 if previous_tag.startswith('I-') else 0) if i > 0 else None
    
    return features_dict


  def generate_classifier(self):
    """
    Returns a trained MaxEnt classifier for the MEMM model on the featurized tokens.
    Use `extract_features_token` to extract features per token.

    Output: 
      classifier: nltk.classify.maxent.MaxentClassifier 
    """
    # YOUR CODE HERE 
    data = []
    for sentence, tags in zip(self.documents, self.labels):
      for i in range(0,len(sentence)):
        if i == 0:
          features = self.extract_features_token(sentence,i,previous_tag=None)
        else:
          features = self.extract_features_token(sentence,i,previous_tag=tags[i-1])
        data.append((features, tags[i]))

    classifier = classify.MaxentClassifier.train(data, max_iter=10)

    return classifier


  def get_trellis_arc(self, predicted_tag, previous_tag, document, i): 
    """
    Returns the trellis arc used by the Viterbi algorithm for the label 
    `predicted_tag` conditioned on the features of the token of `document` at 
    index `i`.
    
    For MEMM, this would be the log classifier output log[P(predicted_tag | features_i)].
  
    Input: 
      predicted_tag: string, predicted tag for token at index `i` in `document`
      previous_tag: string, previous tag for token at index `i` - 1
      document: string
      i: index of the `document` to compute probabilities 
    Output: 
      result: Float
    """
    # YOUR CODE HERE 

    features = self.extract_features_token(document,i,previous_tag)
    prob = self.classifier.prob_classify(features)
    return prob.logprob(sample=predicted_tag)

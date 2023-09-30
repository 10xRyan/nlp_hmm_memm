import numpy as np
from viterbi import viterbi

def flatten_double_lst(lstlst):
   '''
   Returns flattened list version of double nested list (row-major)

   Input: 
    lstlst: List[List[Any]]
   Output:
    lst: List[Any]

   e.g: if lstlst = [[1,2], [3,4]]
        flatten_double_lst(lstlst) returns [1,2,3,4]
   '''
   return [element for lst in lstlst for element in lst]


def format_output_labels(token_labels, token_indices):
    label_dict = {"LOC":[], "MISC":[], "ORG":[], "PER":[]}
    prev_label = 'O'
    start = token_indices[0]
    for idx, label in enumerate(token_labels):
      curr_label = label.split('-')[-1]
      if label.startswith("B-") or (curr_label != prev_label and curr_label != "O"):
        if prev_label != "O":
            label_dict[prev_label].append((start, token_indices[idx-1]))
        start = token_indices[idx]
      elif label == "O" and prev_label != "O":
        label_dict[prev_label].append((start, token_indices[idx-1]))
        start = None

      prev_label = curr_label
    if start is not None:
      label_dict[prev_label].append((start, token_indices[idx-1]))
    return label_dict


def mean_f1(y_pred_dict, y_true_dict):
    F1_lst = []
    for key in y_true_dict:
        num_correct, num_true = 0, 0
        preds = y_pred_dict[key]
        trues = y_true_dict[key]
        for true in trues:
            num_true += 1
            if true in preds:
                num_correct += 1
            else:
                continue
        num_pred = len(preds)
        if num_true != 0:
            if num_pred != 0 and num_correct != 0:
                R = num_correct / num_true
                P = num_correct / num_pred
                F1 = 2*P*R / (P + R)
            else:
                F1 = 0      # either no predictions or no correct predictions
        else:
            continue
        F1_lst.append(F1)
    return np.mean(F1_lst)


def evaluate_model(model, val_set, tags):

  tag = flatten_double_lst(val_set['NER'])
  indices = flatten_double_lst(val_set['index'])

  tag_errors = {}
  tag_totals = {}
  for i in range(len(tags)):
    tag_totals[tags[i]] = 0
    tag_errors[tags[i]] = 0


  output = []
  l = len(val_set['text'])
  for i, sentence in enumerate(val_set['text']):
    v = viterbi(model, sentence ,tags)
    output.append(v)
    for j in range(len(v)):
      tag_totals[v[j]] += 1
      if v[j] != val_set['NER'][i][j]:
        tag_errors[v[j]] += 1

  
  output_tag=flatten_double_lst(output)

  pred = format_output_labels(output_tag, indices)
  actual = format_output_labels(tag,indices)

  print('Tag Errors:', tag_errors)
  print('Tag Totals:', tag_totals)
  tag_error_rate = {}
  for i in tag_errors:
    if tag_totals[i] == 0:
      tag_error_rate[i] = 0
    else:
      tag_error_rate[i] = tag_errors[i]/tag_totals[i]
  
  print('Tag Error Rate:', tag_error_rate)

  print('Score:', mean_f1(pred,actual))
  return mean_f1(pred,actual)


def get_predictions(model, data, tags):
    predictions = []
    total = len(data)
    for i, j in enumerate(data):
        print(f"{i+1}/{total}")
        predictions.append(viterbi(model,j,tags))
    return predictions

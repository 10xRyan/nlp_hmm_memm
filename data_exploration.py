import json
import zipfile
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

def unzip_file(zip_filepath, dest_path):
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(dest_path)
        return True
    except Exception as e:
        return False


def unzip_data(zipTarget, destPath):

    unzip_file(zipTarget, destPath)

    # Get the name of the subdirectory
    sub_dir_name = os.path.splitext(os.path.basename(zipTarget))[0]
    sub_dir_path = os.path.join(destPath, sub_dir_name)

    # Move all files from the subdirectory to the parent directory
    for filename in os.listdir(sub_dir_path):
        shutil.move(os.path.join(sub_dir_path, filename), destPath)

    # Remove the subdirectory
    os.rmdir(sub_dir_path)


def read_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def load_dataset(data_zip_path, dest_path):
    unzip_data(data_zip_path, dest_path)
    training_data = read_json(os.path.join(dest_path, "train.json"))
    validation_data = read_json(os.path.join(dest_path, "val.json"))
    test_data = read_json(os.path.join(dest_path, "test.json"))
    return training_data, validation_data, test_data

def stringify_labeled_doc(text, ner):
    prev_ner = 'O'
    result = ''
    for i in range(len(text)):
      if ner[i] == 'O':
        if prev_ner != 'O':
          result += '] '
          result += text[i]
        else:
          result += ' ' + text[i]
        prev_ner = 'O'        
      # If {something}-{ORG, PER, LOC, MISC}
      elif ner[i].split('-')[1] in { 'ORG', 'PER', 'LOC', 'MISC'}:
        if prev_ner == 'O':
          result += ' [' + ner[i].split('-')[1] + ' '
          result += text[i]
        elif prev_ner == ner[i].split('-')[1]:
          result += ' ' + text[i]
        else:
          result += '] [' + ner[i].split('-')[1] + ' '
          result += text[i]
        prev_ner = ner[i].split('-')[1]        
    
    if prev_ner != 'O':
      result += ']'

    return result.strip()

def validate_ner_sequence(ner):
    for i in range(0,len(ner)-1):
        #Consecutive B-PER
        if ner[i]=="B-PER":
            if ner[i+1]=="B-PER":
                return False
        
        #Consecutive B-LOC
        if ner[i]=="B-LOC":
            if ner[i+1]=="B-LOC":
                return False
        
        #Consecutive B-ORG
        if ner[i]=="B-ORG":
            if ner[i+1]=="B-ORG":
                return False

        #Consecutive B-MISC
        if ner[i]=="B-MISC":
            if ner[i+1]=="B-MISC":
                return False
    
    for i in range(len(ner)):
        if ner[i][0] == 'I':
            tag = ner[i][2:]
            if i == 0 or (ner[i-1] != 'B-' + tag and ner[i-1] != 'I-' + tag):
                return False
            elif ner[i-1] == 'I-' + tag:
                j = i - 1
                while j >= 0 and ner[j][0] != 'B-' + tag:
                    j -= 1
                    if ner[j] != 'I-' + tag:
                        break
                if not (j >= 0 and ner[j] == 'B-' + tag):
                  return False

    return True

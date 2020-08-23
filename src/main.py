import pandas as pd
import re
import os
import time
import random
import numpy as np
import copy

try:
  %tensorflow_version 2.x # enable TF 2.x in Colab
except Exception:
  pass

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import SmoothingFunction
from google.colab import drive
import pickle

import spacy

nlp = spacy.load("en_core_web_sm")
smoothie = SmoothingFunction()
from nltk.translate.bleu_score import corpus_bleu

import config as *

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('Please Sprecify the Data Path')
		exit()
	config = configurations(data_path=sys.argv[1])
	if config.MODE == 'mawps'

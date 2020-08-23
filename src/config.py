import sys, getopt
from nltk.translate.bleu_score import SmoothingFunction
import spacy

class configurations(object):
	def __init__(self, data_path='.', load_model=False, load_model_path=None, MODE='mawps'):
		self.nlp = spacy.load("en_core_web_sm")
		self.smoothie = SmoothingFunction()
		self.drive_root = data_path                # 資料存放的位置
		self.load_model = load_model               # 是否需載入模型
		self.load_model_path = load_model_path     # 載入模型的位置 e.g. "./ckpt/model_{step}" 
		self.store_model_path = "ADL Project"      # 儲存模型的位置
		self.isMapping = True                      # 是否代換數字
		self.MAX_LENGTH = 40
		self.MODE = MODE
		self.BATCH_SIZE = 64
		self.num_layers = 4
		self.dropout_rate = 0.0



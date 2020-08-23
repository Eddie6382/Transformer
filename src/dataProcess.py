import re

'''
Normalize
'''
mode = 'variable'
str_to_num = { 'two':'2', 'three':'3', 'four':'4', 'five':'5', 'six':'6', 'seven':'7', 'eight':'8', 'nine':'9', 'ten':'10', 'once':'1', 'twice':'2', 'half':'0.5' }

def Extract(eq, quan_map, mode):
  eq = re.sub(' ', '', eq)
  if mode == 'unknown':
    variables = []
    for var in re.findall(r'[a-zA-Z_]+', eq):
      if var not in variables:
        variables.append(var)
    return variables
  elif mode == 'variable':
    # substitute the equations
    variables = []
    xy = ['x', 'y', 'z']
    # unknowns
    for var in re.findall(r'(?<!\w)[a-zA-Z_]+(?!\w)', eq):
      if var not in variables:
        variables.append(var)
    for i in range(len(variables)):
      eq = re.sub(rf'(?<!\w){variables[i]}+(?!\w)', xy[i], eq)
    
    # quantities
    eq = re.sub(r'(?<=\.\d)0+', '', eq)
    eq = re.sub(r'\.0+(?!\d)', '', eq)
    for quan, num in quan_map.items():
      eq = re.sub(rf'(?<![\d\.N]){num}(?![\d\.])', quan, eq)
    # print(eq)
    return eq.split('\n')

def genMap(numbers):
  quan_map = {}
  for i in range(len(numbers)):
    quan_map['N'+str(i+1)] = numbers[i]
  return quan_map

def Normalize(question, mode):
    numbers = []
    if mode == 'unknown':
      return question
    elif mode == 'variable':
      question = question.lower()
      for (key, value) in str_to_num.items():
        question = re.sub(key, value, question)
      question = re.sub(r'(?<=\d),(?=\d)', '', question)
      question = re.sub(r'(?<=\.\d)0+', '', question)

      # generate the maop between quantities and numnber
      for number in re.findall(r'(\d+(\,\d+)?(\.\d+)?)', question):
        if number[0] not in numbers:
          tmp = str(number[0])
          tmp = re.sub(r'\.0$', '', tmp)
          tmp = re.sub(',', '', tmp)
          numbers.append(tmp)
      quan_map =  genMap(numbers)
      # print(quan_map)

      # substitute the question sentence
      for quan, num in quan_map.items():
        question = re.sub(rf'(?<![\d\.N]){num}(?!(\d|\.\d+))', quan, question)
      return question, quan_map

def convert_eqn(eqn):
  '''
  Add a space between every character in the equation string.
  Eg: 'x = 23 + 88' becomes 'x =  2 3 + 8 8'
  '''
  elements = list(eqn)
  return ' '.join(elements)

def pklNormalize(inputs_exps, target_exps):
  for i in range(len(input_exps)):
    input_exps[i], quan_map = Normalize(input_exps[i], mode)
    eq = re.sub(r'\s+', '', target_exps[i])
    nor_eq = Extract(eq, quan_map, mode)
    # print(input_exps[i], nor_eq)
    target_exps[i] = ' '.join(list(nor_eq[0]))
    target_exps[i] = re.sub(r'(?<=N)\s(?=\d)', '', target_exps[i])
  return input_exps, target_exps


'''
預處理
'''
def preprocess_input(sentence):
  '''
  For the word problem, convert everything to lowercase, add spaces around all
  punctuations and digits, and remove any extra spaces. 
  '''
  sentence = sentence.lower().strip()
  sentence = re.sub(r"([?.!,’])", r" \1 ", sentence)
  sentence = re.sub(r"(?<![nN])([0-9])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  sentence = sentence.rstrip().strip()
  return sentence

def preprocess_target(sentence):
  '''
  For the equation, convert it to lowercase and remove extra spaces
  '''
  sentence = sentence.lower().strip()
  return sentence

def tokenize(lang):
  '''
  Tokenize the given list of strings and return the tokenized output
  along with the fitted tokenizer.
  '''
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)            # This method creates the vocabulary index based on word frequency, like {'the':1, 'earth':2, .......}
  tensor = lang_tokenizer.texts_to_sequences(lang)
  return tensor, lang_tokenizer

def append_start_end(x,last_int):
  '''
  Add integers for start and end tokens for input/target exps
  '''
  l = []
  l.append(last_int+1)
  l.extend(x)
  l.append(last_int+2)
  return l
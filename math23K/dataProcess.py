import json
import sys
import os
import re
import random
from pprint import pprint
from tqdm import tqdm

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
        xy = ['x', 'y']
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

        # substitution
        eq = re.sub('%', '*0.01', eq)
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

        # generate the map between quantities and numnber
        # print(question)
        for number in  re.findall(r'((\d+(\,\d+)?(\.\d+)?)|\(\d+\/\d+\))', question):
            if number[0] not in numbers:
                tmp = str(number[0])
                tmp = re.sub(r'\.0$', '', tmp)
                tmp = re.sub(',', '', tmp)
                numbers.append(tmp)
        quan_map =  genMap(numbers)

        # substitute the question sentence
        for quan, num in quan_map.items():
            question = re.sub(rf'(?<!([\d\.N]|\/)){num}(?!(\d|\.\d+|\/))', quan, question)
        # print(question)
        return question, quan_map


def createData(problemList):
    content = []
    for (iIndex, sQuestion, lEquations, lSolutions) in problemList:
        # if (type(iIndex) == str and re.search(r'\d+', iIndex)):
        #     iIndex = int(re.search(r'\d+', iIndex).group())
        #     iIndex += 1000
        dictionary = {'iIndex':iIndex, 'sQuestion':sQuestion, 'lEquations':lEquations,'lSolutions':lSolutions}
        content.append(dictionary)
    return content

def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Missing Saving Dir(1), Mode(2) [word, character], ')
        exit(-1)
    Dir = sys.argv[1]
    mode = 'variable'
    style = sys.argv[2]
    problemList = []
    str_to_num = { 'two':'2', 'three':'3', 'four':'4', 'five':'5', 'six':'6', 'seven':'7', 'eight':'8', 'nine':'9', 'ten':'10', 'once':'1', 'twice':'2', 'half':'0.5' }
    input_files = ['math23k_train.json', 'math23k_test.json']

    string = '甲 乙丙 3 人 共同 加工 一批 零件 ， 甲 加工 了 总数 的 40% ， 乙 加工 了 总数 的 (3/8) 还 多 26 个 ， 丙 加工 了 剩下 的 64 个 ． 这 批 零件 一共 有 多少 个 ？'
    print(Normalize(string, 'variable'))

    ID = 10000
    for input_file in input_files:
        data = load_raw_data(input_file)
        for question in data:
            # print('index: ',question['id'])
            ID += 1
            if style == 'word':
                sentence, quan_map = Normalize(question['segmented_text'], mode)
            elif style == 'character':
                sentence = ' '.join(list(question['original_text']))
                sentence = re.sub(r'([\d\.\,\/\)\()])\s(?=[\d\.\,\/\)])', r'\1', sentence)
                # print(sentence)
                sentence, quan_map = Normalize(sentence, mode)
            else:
                print('Error Mode!')
                exit(-1)
            eq = question['equation']
            nor_eq = Extract(eq, quan_map, mode)
            problemList.append((question['id'], sentence, nor_eq, question['ans'] ))
                   
        print(problemList[:10])
    with open(os.path.join(Dir, 'questions.json'), 'w') as json_file:
        json_file.write(json.dumps(createData(problemList), ensure_ascii=False, indent=2))
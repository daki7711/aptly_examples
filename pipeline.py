# import all of the libraries
import os
import time
from glob import glob
import pandas as pd
import openai
import numpy as np
import re
import random

import transformers
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')

BEST_OF = 10

# replace this with your openai api key
openai.api_key = "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"

user_query = "Display the question: 'which is better, MIT or Harvard?' and three answer choices: MIT, Harvard, Not sure. If the user chooses MIT, display the result 'you are a future engineer.' If the user chooses Harvard display the result 'this is MIT app inventor not Harvard app inventor.' If the user chooses Not sure display 'interesting.' Once the user makes a choice, disable all buttons."

# required functions
def get_description(file):
    """
    Extract the description of the code
    """
    with open(file) as f:
        description = f.readline().rstrip()

    return description

def get_code(file):
    with open(file) as f:
        content = f.read()
   
    start = content.index('START\n')
    end = content.index('\nSTOP')
    if start > end:
        raise SyntaxError('Example programs need to be contained between START and STOP markers')
    content = content[content.index('START\n') + 6:content.index('\nSTOP') + 1]

    return content
    
def get_files(filepath):
    """
    Get all text files
    """
    description = get_description(filepath)
    code = get_code(filepath)
    yield {"description": description, "code": code}
  
def get_all_examples():
    # get user root directory
    root_dir = os.path.expanduser("~")

    # path to code repository directory
    code_root = "aptly_examples/examples/"
    code_files = [y for x in os.walk(code_root) for y in glob(os.path.join(x[0], '*.txt'))]
    print("Total number of files:", len(code_files))

    all_examples = []
    for code_file in code_files:
        print(code_file)
        try:
          example = list(get_files(code_file))
          for e in example:
            all_examples.append(e)
        except:
          print("error")
    return all_examples
    
def token(code):
    return nltk.word_tokenize(code)

def embedding(x):
    time.sleep(1)
    return get_embedding(x, engine='code-search-babbage-code-001')
  
def search_functions(df, code_query, n=7):
    embedding_code = get_embedding(code_query, engine='code-search-babbage-text-001')
    df['similarities'] = df.code_embedding.apply(lambda x: cosine_similarity(x, embedding_code))
    res = df.sort_values('similarities', ascending=False).head(n)
    
    return res
  
def bleu(res, rank):
    score = []
    for i in range(len(res['code'])):
      reference = [token(res['code'].iloc[i])]
      candidate = token(res['code'].iloc[rank])
      score.append(sentence_bleu(reference, candidate))
    return score
  
def cosine(res, rank):
  cosine = []
  for i in range(len(res['code_embedding'])):
    cosine.append(cosine_similarity(res['code_embedding'].iloc[rank], res['code_embedding'].iloc[i]))
  return cosine
  
def mrmr(res, rank, emb):
    # embedding : the similarity of the candidate example and the target code
    embedding = res['similarities'].iloc[rank]
    # emb : a full list of mutual information between all the examples and the chosen ones
    # m : The number of chosen examples + 1
    m = len(np.transpose(emb))+1
    # Returning the amount of new information
    # Subtracting the information with the chosen examples from the candidate's information with the target
    return embedding-1/(m-1)*sum(emb[rank])

def generate_program(prompt, max_tokens, best_of):
    print('Prompt length in tokens: ', len(tokenizer(prompt)['input_ids']))
    try:
        completion = openai.Completion.create(
          engine='code-davinci-002',
          prompt=prompt,
          stop='STOP',
          temperature=0.5,
          max_tokens=max_tokens,
          best_of=best_of)
        return completion.choices[0].text
        
    #if the Open API quota is exhausted then we will be getting the OpenAi RateLimitError
    except openai.error.RateLimitError:
        time.sleep(70)
        
        completion = openai.Completion.create(
          engine='code-davinci-002',
          prompt=prompt,
          stop='STOP',
          temperature=0.5,
          max_tokens=max_tokens,
          best_of=best_of)
        return completion.choices[0].text

def pipeline(res, method, tkn, order):
    print(order, method, tkn)
    
    if method == 'random':
        res2 = res.copy()
        res2 = res2.sample(frac = 1)
        itr = 0
        chosen = []
        
        length = len(tokenizer('%'+user_query+'%\n')['input_ids'])
        while length<tkn:
            text = '%' + res2['description'].iloc[itr] + '%\n'
            text += res2['code'].iloc[itr]
            text += 'STOP\n'
            if length+len(tokenizer(text)["input_ids"])<=tkn:
                length+=len(tokenizer(text)["input_ids"])
                chosen.append(itr)
                itr+=1
            else:
                break
        
        res3 = res2
    elif method == 'size':
        itr = 0
        chosen = []
        sizes = []

        ## Sorting the examples code by sizes
        for i in res['code']:
          sizes.append(len(i))
        res2 = res.copy()
        res2['size'] = sizes
        res2 = res2.sort_values('size')

        ## Getting all the examples that fit into the limited tokens
        ## Retrieving ''chosen'' list
        length = len(tokenizer('%'+user_query+'%\n')['input_ids'])
        while length<tkn:
          text = '%' + res2['description'].iloc[itr] + '%\n'
          text += res2['code'].iloc[itr]
          text += 'STOP\n'
          if length+len(tokenizer(text)["input_ids"])<=tkn:
            length+=len(tokenizer(text)["input_ids"])
            chosen.append(itr)
            itr+=1
          else:
            break
          
        res3 = res2
  
    elif method == 'embeddings':
        chosen = []
        itr = 0
        ## Getting all the examples that fit into the limited tokens
        ## Retrieving ''chosen'' list
        length = len(tokenizer('%'+user_query+'%\n')['input_ids'])
        while length<tkn:
          text = '%' + res['description'].iloc[itr] + '%\n'
          text += res['code'].iloc[itr]
          text += 'STOP\n'
          if length+len(tokenizer(text)["input_ids"])<=tkn:
            length+=len(tokenizer(text)["input_ids"])
            chosen.append(itr)
            itr+=1
          else:
            break
            
        res3=res
    
    elif method == 'mrmr':
        chosen = [0]
        cosines1 = []
        itr = 0
        code = res['code'].iloc[0]
        
        ## Getting all the examples that fit into the limited tokens
        ## Retrieving ''chosen'' list
        length = len(tokenizer(code+'STOP\n')['input_ids'])+len(tokenizer('%'+user_query+'%\n')['input_ids'])+len(tokenizer('%'+res['description'].iloc[0]+'%\n')['input_ids'])
        while length<tkn:
            cosines1 += [cosine(res,chosen[itr])]
            examine = list(range(len(res['code'])))
            high = -20
            hterm = 0
            # examine : the list of candidates
            for i in chosen:
                examine.remove(i)
            
            # The process of finding the candidate with maximum information
            for ind in examine:
                # cosines : a full list of mutual information between all the examples and the chosen ones
                cosines = np.transpose(cosines1)
                val = mrmr(res, ind, cosines)
                if val > high:
                    high = val
                    hterm = ind
                
            itr+=1
            text = '%' + res['description'].iloc[hterm] + '%\n'
            text += res['code'].iloc[hterm]
            text += 'STOP\n'
            if length+len(tokenizer(text)['input_ids'])<=tkn:
                length+=len(tokenizer(text)['input_ids'])
                chosen.append(hterm)
            else:
                break;
        res3= res
    
    
    full_prompt = ""

    if order == 'bottom':
        chosen = chosen[::-1]
    elif order == 'random':
        random.shuffle(chosen)
    else:
        chosen = chosen

    for r in res3.iloc[chosen].iterrows():
        full_prompt += '%' + r[1].description + '%\n'
        full_prompt += r[1].code
        full_prompt += 'STOP\n'

    full_prompt += '%'+user_query+'%\n'
    max_tokens = 1500
    aptly_program = generate_program(full_prompt, max_tokens, BEST_OF)
    return aptly_program

def run(itr, res):
    path = "result/" + str(itr)
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory '% s' created" % path)
    
    orders = ['top', 'bottom', 'random']
    methods = ['random', 'size', 'embeddings', 'mrmr']
    tkns = [500, 1000, 1500, 2000, 2500]
    
    df_result = pd.DataFrame(columns = ['iteration', 'order', 'method', 'tkn', 'code', 'time'])
    ind = 0
    
    path = "result/" + str(itr) + "/method"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory '% s' created" % path)
    
    order = 'top'
    tkn = 1500
    # BY method
    for method in methods:
        st = time.time()
        title = 'top_' + method + '_1500'
        with open(path + "/" + title + ".txt", 'w') as f:
            result = pipeline(res, method, 1500, 'top')
            f.write(result)

        ind += 1
        et = time.time()
        print(et-st)
        df_result.loc[ind] = [itr, order, method, tkn,result, et-st]
        time.sleep(10)
        
    path = "result/" + str(itr) + "/tokens"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory '% s' created" % path)
    
    order = 'top'
    method = 'mrmr'
    # By Token size
    for tkn in tkns:
        st = time.time()
        title = 'top_' + 'mrmr_' + str(tkn)
        with open(path + "/" + title + ".txt", 'w') as f:
            result = pipeline(res,'mrmr', tkn, 'top')
            f.write(result)
        ind += 1
        et = time.time()
        print(et-st)
        df_result.loc[ind] = [itr, order, method, tkn,result, et-st]
        time.sleep(10)
    
    path = "result/" + str(itr) + "/order"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory '% s' created" % path)
    method = 'mrmr'
    tkn = 1500
    # By Ordering
    for order in orders:
        st = time.time()
        title = order + '_mrmr' + '_1500'
        with open(path + "/" + title + ".txt", 'w') as f:
            result = pipeline(res, 'mrmr', 1500, order)
            f.write(result)
        ind += 1
        et = time.time()
        print(et-st)
        df_result.loc[ind] = [itr, order, method, tkn,result, et-st]
        time.sleep(10)

def main():
    if os.path.exists('example.json'):
        #df = pd.read_csv('example.csv', dtype= {'description': 'str', 'code': 'str', 'code_embedding': 'str'})
        #df['code_embedding'] = df['code_embedding'].apply(lambda x: np.array(x))
        df = pd.read_json('example.json')
    else:
        all_examples = get_all_examples()
        
        df = pd.DataFrame(all_examples)
        df['code_embedding'] = df["code"].apply(lambda x: embedding(x))
        df.to_csv('example.csv')
        df.to_json('example.json')
    
    #print(len(df['code_embedding'][1]))
    res = search_functions(df, user_query, n=len(df['code']))

    path = "result"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory '% s' created" % path)

    for itr in range(10):
        run(itr, res)

if __name__ == "__main__":
    main()

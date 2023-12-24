import os
os.environ["HF_DATASETS_CACHE"] = "/mount/arbeitsdaten/cik/users/jonathan/.cache/huggingface/datasets"
import spacy
import itertools
from lemminflect import *

nlp = spacy.load('en_core_web_lg')

def getAllPossibleLemmas(word):
    return [l[0] for l in getAllLemmas(word).values()]

# returns position of sub list in list, or [-1, -1] if not found
def get_word_position(word, sentence):
    ln = len(word)
    for i in range(len(sentence) - ln + 1):
        if sentence[i:i+ln] == word:
            return [i, i+ln-1]
    return [-1, -1]

# constructs a list of possible usages for a word, either as a list of tokens or as a string
def construct_multi_word(token_list, seperator, as_list):
    word_usages = [''.join(token_list) if not as_list else token_list]
    word_usages.extend([' '.join(token_list) if not as_list else token_list])

    for s in seperator:
        usage = f"|{s}|".join(token_list).split('|') # create a list of tokens
        
        # if we want a string
        if not as_list:
            new_usage = ""
            for token in usage:
                new_usage += token
            usage = new_usage
        
        word_usages.append(usage) # append to possible usages

        # avoide duplicates
        if len(token_list) == 1: 
            break

    return word_usages

def find_substring(sentence, word_usage):   
    candidate_indieces = []

    # find all occurences of word usage in sentence
    if sentence.startswith(word_usage):
        candidate_indieces = [0]
    
    for i in range(1, len(sentence) - len(word_usage) + 1):
        if sentence.startswith(word_usage, i):
            if not sentence[i-1].isalpha():
                candidate_indieces.append(i)

    if len(candidate_indieces) == 0:
        return [-1, -1]

    sentence_index_1 = -1
    
    for i in candidate_indieces: # for all possible occurences (except the last one)
        if i + len(word_usage) < len(sentence): # if word usage is not at the end of the sentence
            if not sentence[i + len(word_usage)].isalpha(): # if character after word usage is not a [a-z]
                sentence_index_1 = i
                break

    if sentence_index_1 == -1: # if no occurence was found
        if sentence[-len(word_usage):] == word_usage: # if word usage is at the end of the sentence
            sentence_index_1 = candidate_indieces[-1]

    sentence_index_2 = sentence_index_1 + len(word_usage)
    if sentence[sentence_index_1:sentence_index_2] != word_usage:
        return [-1, -1]

    return [sentence_index_1, sentence_index_2]

# returns a lemma that is eeparated by '_' instead of ' ' or '-' or '/'
def standardize_lemma(lemma):
    return lemma.replace('-', '_').replace(' ', '_').replace('/', '_').replace(', ', '_').replace(' , ', '_')

def get_indieces(sentence, lemma):
    # if no inflections are needed
    if get_character_index(sentence, lemma)["lemma"] != [-1, -1]:
        #print(f"return A")
        return get_character_index(sentence, lemma)
    
        
    # build inflections
    possible_lemmas = []

    # append lemminflect lemmas of whole lemma
    for l in getAllLemmas(lemma).values():
        for v in l:
            possible_lemmas.append(v)
        
    #possible_lemmas.extend(construct_lemmas(lemma))

    possible_lemmas = list(set(possible_lemmas))

    lemma_inflections = []
    lemma_parts = standardize_lemma(lemma).split('_') # get parts of lemma
    for l in lemma_parts:  # for all lemma parts
        if l == '':
            continue

        unique_inflections = {l} # add original lemma part
        for t in list(getAllInflections(l).values()):
            for v in t:
                unique_inflections.add(v)

        lemma_inflections.append(unique_inflections)

    combinations = list(itertools.product(*lemma_inflections))
    unique_combinations = list(set(combinations))

    for uc in unique_combinations:
        possible_lemmas.extend(construct_multi_word(uc, ['-'], as_list=False))

    for l in possible_lemmas:
        indieces = get_character_index(sentence, l)
        if indieces["word"] != [-1, -1]:
            #print(f"found [{l}] in \"{sentence}\" at {indieces}")
            #print(f"return B")
            return indieces

    #print(f"return C")
    return get_character_index(sentence, lemma)

# calls get_word_position for all possible lemmas breaks if lemma is found
def search_word_usages(sentence, lemmas):
    for l in lemmas: # for all possible lemmas
        lemma_index = get_word_position(l, sentence) 
        if lemma_index != [-1, -1]:
            break
    return lemma_index

def construct_lemmas(lemma):
    lemma_doc = nlp(lemma, disable=['ner'])
    lemma_parts = standardize_lemma(lemma).split('_')

    possible_lemmas = []

    spacy_lemmas = [l.lemma_ for l in lemma_doc] # append spacy lemmatization

    lemminflect_lemmas = [l._.lemma() for l in lemma_doc] # append lemminflect lemmatization

    possible_lemmas.extend(construct_multi_word(spacy_lemmas, ['-'], as_list=False))
    possible_lemmas.extend(construct_multi_word(lemminflect_lemmas, ['-'], as_list=False))

    possible_lemmas = list(set(possible_lemmas))
    #print(f"possible lemmas: {possible_lemmas}")

    # split, lemmatize and reconstruct lemma
    spacy_parts = []
    lemminflect_parts = []

    for lp in lemma_parts:
        lp_doc = nlp(lp, disable=['ner'])
        spacy_parts.append([l.lemma_ for l in lp_doc][0]) # append spacy lemmatization
        lemminflect_parts.append([l._.lemma() for l in lp_doc][0]) # append lemminflect lemmatization

    possible_lemmas.extend(construct_multi_word(spacy_parts, ['-', '/', '_'], as_list=False))
    possible_lemmas.extend(construct_multi_word(lemminflect_parts, ['-', '/', '_'], as_list=False))

    return possible_lemmas

# get character index of word usage and lemma in sentence
def get_character_index(sentence, lemma):
    # parse sentence and get tokens and lemmas
    doc = nlp(sentence, disable=['ner'])
    tokenized = [token.text for token in doc]
    lemmatized = [token.lemma_.lower() for token in doc]
    lemma_parts = standardize_lemma(lemma).split('_')

    
    if '_' in lemma and '-' in lemma:
        multiwords = lemma.split('_')
        indieces = []
        for mw in multiwords:
            #print(f"searching for [{mw}] in \"{sentence}\"")
            indieces.append(get_indieces(sentence, mw))

        continueing = True

        #print(f"indieces: {indieces}")
        for i in range(len(indieces) - 1):
            if indieces[i]["word"][-1] != indieces[i+1]["word"][0] - 1:
                continueing = False
    
        if indieces[-2]["word"][-1] != indieces[-1]["word"][0] - 1:
            continueing = False

        lemma_index = [indieces[0]["lemma"][0], indieces[-1]["word"][-1]]
        word_index = [indieces[0]["word"][0], indieces[-1]["word"][-1]]

        if continueing:
            return {"lemma": lemma_index, "word": word_index}
    
    if lemma in sentence: # if lemma is in sentence in its original form
        #print(f"checking if [{lemma}] in original form in \"{sentence}\"")
        if lemma in tokenized: # if lemma is not seperated by '_' or '-' or '/'
            lemma_index = [tokenized.index(lemma), tokenized.index(lemma)] # get index of lemma in tokenized sentence
        else: # lemma is seperated by '_' or '-' or '/'
            possible_lemmas = construct_multi_word(lemma_parts, ['-', '/', '_'], as_list=True)
            lemma_index = search_word_usages(tokenized, possible_lemmas)
    else:
        possible_lemmas = construct_lemmas(lemma)
        lemma_index = search_word_usages(lemmatized, possible_lemmas)

    if lemma_index == [-1, -1]:
        return {"lemma": [-1, -1], "word": [-1, -1]}
    
    # construct possible word usages
    token_list = tokenized[lemma_index[0]:lemma_index[-1] + 1]
    word_usages = construct_multi_word(token_list, ['-', '/', ', ', ' , ', ' '], as_list=False)

    #if any (w in sentence for w in word_usages):
    for w in word_usages:
        if w in sentence:
            word_usage = w
            return {"lemma": lemma_index, "word": find_substring(sentence, word_usage)}

    return {"lemma": [-1, -1], "word": [-1, -1]}
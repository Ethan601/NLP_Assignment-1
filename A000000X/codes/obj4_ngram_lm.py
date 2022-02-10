'''
    NUS CS4248 Assignment 1 - Objective 4 (n-gram Language Model)

    Class NgramLM for handling Objective 4

    Important: please strictly comply with the input/output formats for
               the methods of generate_word & generate_text & perplexity, 
               as we will call them in testing
    
    Sentences for Task 4-B:
    1) "They had now entered a beautiful walk by"
    2) "The snakes entered a beautiful walk by the buildings."
    3) "They had now entered a walk by"
'''

###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

import re
import random, math
import numpy as np
from collections import defaultdict


class NgramLM(object):

    def __init__(self, n, k):
        '''
            Initialize your n-gram LM class

            Parameters:
                n (int) : order of the n-gram model
                k (float) : smoothing hyperparameter

        '''
        # Initialise other variables as necessary
        # TODO Write your code here
        self.n = n
        self.k = k
        self.word_count_dict = {}
        self.sos = '<s>'
        self.eos = '</s>'
        self.ngram_counter = defaultdict(int)
        self.context = {}
        self.tokens = []
        self.text = ""
        self.vocabulary = set()
        self.sentence = []
        self.start_context = []

    def update_corpus(self, text):
        ''' Updates the n-grams corpus based on text '''
        # TODO Write your code here
        split_text = re.split(r'\W+', text)
        self.tokens = [w for w in split_text if w]

        if len(self.tokens) < self.n:
            return

        ngrams = self.ngrams()
        for ngram in ngrams:
            self.ngram_counter[ngram] += 1.0
            prev_words, target_word = ngram
            if prev_words in self.context:
                self.context[prev_words].append(target_word)
            else:
                self.context[prev_words] = [target_word]

        # split sentences
        pattern = re.compile(r'[?!.]\s+')
        for each in re.split(pattern, text):
            temp = each.lower()
            self.sentence.append([w for w in re.split(r'\W+', temp) if w])

    def read_file(self, path):
        ''' Read the file and update the corpus  '''
        # TODO Write your code here
        text_corpus = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            self.text = f.read()
        self.update_corpus(self.text)

        self.text = self.text.encode("ascii", "ignore").decode()
        self.text = re.sub(r'\s+', ' ', self.text)
        self.text = self.text.lower()

        self.update_corpus(self.text)

    def ngrams(self):
        ''' Returns ngrams of the text as list of pairs - [(sequence context, word)] '''
        # TODO Write your code here
        self.tokens = (self.n - 1) * [self.sos] + self.tokens + (self.n - 1) * [self.eos]
        l = [(tuple([self.tokens[i - p - 1] for p in reversed(range(self.n - 1))]), self.tokens[i]) for i in
             range(self.n - 1, len(self.tokens))]
        return l

    def add_padding(self):
        '''  Returns padded text '''
        # TODO Write your code here
        # Use '~' as your padding symbol
        return (self.n - (len(self.start_context) + 1)) * ['~']

    def get_vocabulary(self):
        ''' Returns the vocabulary as set of words '''
        # TODO Write your code here
        self.vocabulary = self.vocabulary.union(set(self.tokens))
        return self.vocabulary

    def get_next_word_probability(self, text, word):
        ''' Returns the probability of word appearing after specified text '''
        # TODO Write your code here
        try:
            count_of_token = self.ngram_counter[(text, word)]
            count_of_context = float(len(self.context[text]))
            result = (count_of_token + self.k) / (count_of_context + self.k * len(self.get_vocabulary()))
        except KeyError:
            result = 0.0
        return result

    def generate_word(self, text):
        '''
        Returns a random word based on the specified text and n-grams learned
        by the model
        [In] string (short text)
        [Out] string (word)
        '''
        # TODO Write your code here
        if type(text) == str:
            print('str')
            split_string = re.split(r'\W+', text)
            if len(split_string) < (self.n - 1):
                text_tuple = tuple((self.n - (len(split_string) + 1)) * ['~'] + split_string)
            else:
                text_tuple = tuple(split_string[-(self.n - 1):])
        else:
            text_tuple = text

        tokens_of_interest = self.context[text_tuple]
        # print(type(tokens_of_interest[0])) # list
        token_prob = np.array([self.get_next_word_probability(text_tuple, token) for token in tokens_of_interest])

        return np.random.choice(tokens_of_interest, 1, p=(token_prob / sum(token_prob)))[0]

    def generate_text(self, length):
        ''' Returns text of the specified length based on the learned model 
        [In] int (length: number of tokens)
        [Out] string (text)
        '''
        # TODO Write your code here
        context_queue = []
        self.start_context = random.choice(self.sentence)

        if len(self.start_context) == (self.n - 1):
            context_queue = self.start_context.copy()
            result = self.start_context
        elif len(self.start_context) < (self.n - 1):
            context_queue = self.add_padding() + self.start_context.copy()
            result = self.start_context.copy()
        elif len(self.start_context) > (self.n - 1):
            # context_queue = self.start_context[-(self.n - 1):].copy()
            context_queue = self.start_context[:(self.n - 1)].copy()
            result = self.start_context[:(self.n - 1)]

        # generate words
        for _ in range(length - self.n + 1):
            obj = self.generate_word(tuple(context_queue))
            result.append(obj)
            context_queue.pop(0)
            if obj == self.eos:
                return ' '.join(result[:-1])
            else:
                context_queue.append(obj)
        # Fallback if we predict more than token_count tokens
        return ' '.join(result)


    def perplexity(self, text):
        '''
        Returns the perplexity of text based on learned model
        [In] string (a short text)
        [Out] float (perplexity)

        Hint: To avoid numerical underflow, add logs instead of multiplying probabilities.
        Also handle the case when the LM assigns zero probabilities.
        '''
        # TODO Write your code here
        token_list = re.split(r'\W+', text.lower())

        if len(token_list) < (self.n - 1):
            text = (self.n - (len(token_list) + 1)) * ['~'] + token_list
        else:
            text = token_list[:(self.n - 1)]

        log_prob = 0
        # print(text)
        # print(token_list[(self.n - 1):])
        for word in token_list[(self.n - 1):]:
            p = self.get_next_word_probability(tuple(text), word)
            if p == 0:
                return 0
            log_prob += math.log(p, 10)
            text.pop(0)
            text.append(word)

        return log_prob ** (-1 / len(token_list)) ** 10


# res = NgramLM(2, 1)
# res.read_file("../data/Pride_and_Prejudice.txt")
# test_text = "They had now entered a walk by"
# for i in range(6):
#     test_text = test_text + ' ' + res.generate_word(test_text)
#     print(test_text)
# print(res.perplexity(test_text))

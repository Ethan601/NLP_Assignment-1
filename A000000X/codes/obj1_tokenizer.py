'''
    NUS CS4248 Assignment 1 - Objective 1 (Tokenization, Zipf's Law)

    Class Tokenizer for handling Objective 1

    Important: please strictly comply with the input/output formats for
               the method of tokenize_sentence, as we will call it in testing
'''
###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

import matplotlib.pyplot as plt  # Requires matplotlib to create plots.
import numpy as np  # Requires numpy to represent the numbers
import re
# from nltk.tokenize import regexp_tokenize
import math

def draw_plot(r, f, imgname):
    # Data for plotting
    x = np.asarray(r)
    y = np.asarray(f)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Rank (log)', ylabel='Frequency (log)',
           title='Word Frequency v.s. Rank (log)')
    ax.grid()
    fig.savefig(f"../plots/{imgname}")
    plt.show()

try:
    import subprocess
    import sys
    from bpe import Encoder
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'bpe'])
    from bpe import Encoder


class Tokenizer:

    def __init__(self, path, bpe=False, lowercase=True):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            self.text = f.read()

        self.bpe = bpe
        self.lowercase = lowercase

    def tokenize(self):
        ''' Returns/Saves a set of word tokens for the loaded textual file

        For the default setting, make sure you consider cases of:
        1) words ending with punctuation (e.g., 'hiking.' ——> ['hiking', '.']);
        2) numbers (e.g., '1/2', '12.5')
        3) possessive case (e.g., "Elle's book" ——> ["Elle's", "book"])

        For the bpe setting,
        1) tune the number of iterations so the vocab size will be close to the 
        default one's
        2) during merge, for sub-sequences of the same frequency, break the tie 
        with left-to-right byte order precedence
        '''
        # TODO Modify the code here
        if self.lowercase:
            self.text = self.text.lower()

        if not self.bpe:
            pattern = r"\w+[']\w*|\d+[./]\d+|[\w-]+|\d+|[,.;:!?]"
            # return regexp_tokenize(self.text, pattern)
            return re.findall(pattern, self.text)
        elif self.bpe:
            encoder = Encoder(7000, pct_bpe=0.2)
            encoder.fit(self.text.split('\n'))
            return encoder.tokenize(self.text)

    def tokenize_sentence(self, sentence):
        '''
        To verify your implementation, we will test this method by 
        input a sentence specified by us.  
        Please return the list of tokens as the result of tokenization.

        E.g. basic tokenizer (default setting)
        [In] sentence="I give 1/2 of the apple to my ten-year-old sister."
        [Out] return ['i', 'give', '1/2', 'of', 'the', 'apple', 'to', 'my', 'ten-year-old', 'sister', '.']
        
        PS: For BPE, you may need to fix the vocab before tokenizing
            the input sentence
        '''
        # TODO Modify the code here
        if self.lowercase:
            sentence = sentence.lower()

        if not self.bpe:
            pattern = r"\w+[']\w*|\d+[./]\d+|[\w-]+|\d+|[,.;:]"
            # return regexp_tokenize(sentence, pattern)
            return re.findall(pattern, self.text)
        elif self.bpe:
            encoder = Encoder(7000, pct_bpe=0.2)
            encoder.fit(sentence.split('\n'))  # or self.text??????????
            return encoder.tokenize(sentence)

    def plot_word_frequency(self):
        '''
        Plot relative frequency versus rank of word to check
        Zipf's law
        You may want to use matplotlib and the function shown 
        above to create plots
        Relative frequency f = Number of times the word occurs /
                                Total number of word tokens
        Rank r = Index of the word according to word occurence list
        '''
        # TODO Modify the code here
        word_frequency = {}
        tokens = self.tokenize()

        for each in tokens:
            if each not in word_frequency.keys():
                word_frequency[each] = 1
            else:
                word_frequency[each] += 1

        token_len = len(self.tokenize())

        word_freq_sorted = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
        frequency = [math.log(each[1] / token_len, 10) for each in word_freq_sorted]
        rank = [math.log(each, 10) for each in range(1, len(frequency) + 1)]
        if not self.bpe:
            draw_plot(rank, frequency, '1-1-A')
        elif self.bpe:
            draw_plot(rank, frequency, '1-1-B')

# res = Tokenizer('../data/Pride_and_Prejudice.txt',True,True )
# res = res.plot_word_frequency()
# res = res.tokenize()
# res = Tokenizer()
# res = res.tokenize_sentence("I give 1/2 of the apple to my ten-year-old sister.")
# print(res)

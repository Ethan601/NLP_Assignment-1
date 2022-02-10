'''
    NUS CS4248 Assignment 1 - Objective 3 (Regular Expression, Sentiment Analysis)

    Class Tokenizer for handling Objective 3

    Important: please strictly comply with the input/output formats for
               the methods of process_text & classify_sentiment, 
               as we will call them in testing
    
    Sentiment Labels: 1 (positive); -1 (negative); 0 (neutral)
'''

###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

import matplotlib.pyplot as plt     # Requires matplotlib to create plots.
import numpy as np    # Requires numpy to represent the numbers

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


import re
import math
import collections

import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))

def cal_accu(preds, golds):
    compare = [1 if p == g else 0 for p,g in zip(preds, golds)]
    return sum(compare) / len(compare)


class SentimentClassifier:
    def __init__(self, path, emoticon_dir, lexicon_dir):
        ## Load emoticon list
        with open(emoticon_dir, 'r', encoding='utf-8') as f:
            self.emoticons = f.read().strip().split('\n')   # emoticon list
        
        ## Load lexicon sentiment list
        with open(lexicon_dir, 'r', encoding='utf-8') as f:
            data = [d.strip().split('\t') for d in f.read().strip().split('\n')]
        self.lexicon_sentiment = {d[0]: int(d[1]) for d in data}    # lexicon sentiment list

        ## Load tweet corpus
        raw_tweets = json_load(path)   # raw data (labeled) of tweets
        # TODO Modify the code here
        self.id = []
        self.review = []
        self.label = []
        self.token = []
        self.emoticon_ls = []

        for each in raw_tweets:
            self.id.append(each['id'])
            self.review.append(each['review'])
            self.label.append(each['label'])

        for each in self.review:
            temp = self.process_text(each)
            self.token.append(temp)
            if len(temp['emoticons']) != 0:
                self.emoticon_ls.append(temp['emoticons'][0])

        word_pred = []
        emo_pred = []
        for each in self.token:
            word_pred.append(self.classify_sentiment(each,threshold=0.1,use_emoticon=False,ratio=0.5))
            emo_pred.append(self.classify_sentiment(each,threshold=0.1,use_emoticon=True,ratio=0.5))

        print("word sentiment prediction accuracy: " + str(cal_accu(word_pred, self.label)))
        print("emoticon sentiment prediction accuracy: " + str(cal_accu(emo_pred, self.label)))
    
    def judge_emoticon(self, word):
        '''Determine whether a word(token) is an emoticon'''
        # TODO Modify the code here
        # temp = "'"
        # emo_length = len(self.emoticons)
        # count = 0
        #
        # for each in self.emoticons:
        #     count += 1
        #     if count == emo_length:
        #         temp = temp + re.escape(each)
        #         break
        #     temp = temp + re.escape(each) + "|"
        #
        # temp = temp + "'"
        #
        # pattern_emoticon = temp
        #
        # return len(re.findall(pattern_emoticon, word)) != 0
        if word in self.emoticons:
            return True
        else:
            return False


    def process_text(self, text):
        '''Tokenization & Clean & Extract Emoticons on the input text
        Please also clean the tweet data by removing some noise (i.e. email, weblink).
        
        You need to extract the emoticons and distinguish them from other tokens,
        as in the basic implementation of classify_sentiment, only the non-emoticon 
        tokens are utilized.

        [In] original text of a tweet
        [Out] a sample of the review displayed in dict
        E.g.
        [In] text='I like it. :)'
        [Out] return {'raw': 'I like it. :)', 'text': ['I', 'like', 'it', '.'], 'emoticons': [':)']}
        '''
        # TODO Modify the code here
        pattern_email = r'^(?!\S*@)\S*$'
        pattern_web = r'^(?!\S*http:)\S*$'
        pattern_punc = r'^(?!\S*[.,?!])\S*$'
        pattern_WordPuncSplitter = r"\w+[']\w*|\d+[./]\d+|[\w-]+|\d+|[,.;:]"

        temp = text.split()
        token_text = []
        token_emoticons = []

        for i in temp:
            if re.findall(pattern_email, i):
                if re.findall(pattern_web, i):
                    temp2 = re.findall(pattern_punc, i)     # select word w/o punctuations
                    if temp2:                               # word w/o punctuations
                        if self.judge_emoticon(i):          # check if word is emoticon
                            token_emoticons.append(i)
                        else:
                            token_text.append(i)
                    else:                                   # word w/ punctuations
                        for each in re.findall(pattern_WordPuncSplitter, i):
                            token_text.append(each)
                else:
                    continue
            else:
                continue

        return {'raw': text, 'text': token_text, 'emoticons': token_emoticons}
    
    def classify_sentiment(self, sample, threshold=0.1, 
                           use_emoticon=False, ratio=0.5):
        '''Utilize lexicon sentiment (and emoticon) for sentiment analysis

        sample: the input tweet (with tokens, emoticons, tokens excluding emoticons)
        threshold: threshold to decide whether to choose 1/-1 or 0
        use_emoticon: whether to use emoticons for sentiment analysis
        ratio: weights of emoticons when making the final decision

        E.g.
        [In] 
        sample={'raw': 'I like it. :)', 'text': ['I', 'like', 'it', '.'], 'emoticons': [':)']}
        threshold=0.1  use_emoticon=True  ratio=0.25
        [Out] return 1
        '''
        ## Get the lexicon sentiment for each non-emoticon token
        word_sentiment = []
        for word in sample['text']:
            if word in self.lexicon_sentiment:
                if self.lexicon_sentiment[word] == 1:
                    word_sentiment.append(1)
                elif self.lexicon_sentiment[word] == -1:
                    word_sentiment.append(-1)
                else:
                    word_sentiment.append(0)
            else:
                word_sentiment.append(0)

        assert len(sample['text']) == len(word_sentiment)
        
        ## Calculate the avg score as polarity
        lx_label = sum(word_sentiment) / len(word_sentiment)

        ## Whether to utilize the information from emoticons
        if use_emoticon:
            emo_sentiment = []
            ###################################################
            ### Utilize emoticon to improve ###################
            ##           TODO Modify the code here           ##
            ###################################################
            sentiment = {}
            posRegex = r"(:(\)|D|P|p)|(-(\)|D))|(o\)))|(;(\)|-\)|D))|(8\))|(<3)|=(\)|D)|\((:|8)"
            neuRegex = r"8:"
            negRegex = r"(:(\(|\/)|(-(\(|\/)))|(\)(;|:))|((;|=)\()"

            for each in self.emoticons:
                if len(re.findall(posRegex, each)) != 0:
                    sentiment[each] = 1
                elif len(re.findall(neuRegex, each)) != 0:
                    sentiment[each] = 0
                elif len(re.findall(negRegex, each)) != 0:
                    sentiment[each] = -1

            for word in sample['emoticons']:
                if word in sentiment.keys():
                    if sentiment[word] == 1:
                        emo_sentiment.append(1)
                    elif sentiment[word] == -1:
                        emo_sentiment.append(-1)
                    else:
                        emo_sentiment.append(0)
                else:
                    emo_sentiment.append(0)

            assert len(sample['emoticons']) == len(emo_sentiment)

            if len(emo_sentiment) == 0:
                label = lx_label
            else:
                em_label = sum(emo_sentiment) / len(emo_sentiment)
                label = em_label * ratio + lx_label * (1 - ratio)
        else:
            label = lx_label

        ## Generate the discrete labels using the threshold 
        label = 1 if label > threshold else label
        label = -1 if label < -threshold else label
        label = 0 if label not in [-1, 1] else label

        return label

    def plot_emoticon_frequency(self):
        '''
        Plot relative frequency versus rank of emoticon to check
        Zipf's law
        You may want to use matplotlib and the function shown 
        above to create plots
        Relative frequency f = Number of times the emoticon occurs /
                                Total number of emoticon tokens
        Rank r = Index of the emoticon according to emoticon occurence list
        '''
        # TODO Modify the code here
        word_frequency = {}

        for each in self.emoticon_ls:
            if each not in word_frequency.keys():
                word_frequency[each] = 1
            else:
                word_frequency[each] += 1

        emoticon_num = len(self.emoticon_ls)

        word_freq_sorted = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
        frequency = [math.log(each[1] / emoticon_num, 10) for each in word_freq_sorted]
        rank = [math.log(each, 10) for each in range(1, len(frequency) + 1)]

        draw_plot(rank, frequency, '1-3-A')

# res = SentimentClassifier('../data/labeled_tweets.json', '../data/emoticons.txt', '../data/lexicon_sentiment.txt')
# res.plot_emoticon_frequency()



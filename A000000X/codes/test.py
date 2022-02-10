import random
import re
# import nltk
# nltk.download('punkt')
import numpy as np
# from nltk.tokenize import regexp_tokenize
# from nltk.tokenize import word_tokenize

# pattern = r'(\b(\S)[\S]*\2\b)'
# pattern = r'(?=(^(\S)[\S]*\2)$)(?=(^.*(\S)(\S)\5\4.*$))'
# pattern = r'((\S)(\S)\3\2)'
# words = "otto trillion xxxx -++- abcjklsdfsd".split()
# ls = "otto trillion xxxx -++- -+-+ abcjklsdfsd ".split()
# text = "abcd aba cbc, a -++- -+-+  01230"

# words = re.findall(pattern, 'sd----fabbabba')
# for each in ls:
#     word = re.fullmatch(pattern, each)
#     if word is None:
#         continue
#     elif word is not None:
#         words.append(word.group())
# print(words)
# print(len(words))
# [\s]+[(_;/{`~*-]+|[)_;}`~*-]+[\s]|\s+|\W{2,}|[,.]\s

# print(regexp_tokenize("each", pattern))
# print(type(word_tokenize("I give 1/2 of the apple to my ten-year-old sister.")))

# tokens = []
# word_frequency = {}
# for each in tokens:
#     if each not in word_frequency.keys():
#         word_frequency[each] = 1
#     else:
#         word_frequency[each] += 1
# print(tokens[0])
# test: r':\)|:\(|:\-\)|;\)|\);|\):|:D|\(8|<3|;\-\)|:/|=\)|:\-\(|8:|8\)|\(:|:P|;D|:o\)|=\(|:p|:\-/|:\-D|;\(|=D'
text = [{'id': 1, 'review': "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D", 'label': -1},
        {'id': 2, 'review': '@amyg0716 thats really sad  i wolud hate that! but i had choco milk earlier =D lol', 'label': -1},
        {'id': 3, 'review': "went to bed at 8pm and now wide awake at 4 am. I don't have to be to work until 8:30. This is going to be a long day ", 'label': -1}]
review = []

# for each in text:
#     review.append(each['review'])

# pattern = r"\w+[']\w*|\d+[./]\d+|[\w-]+|\d+|[,.;:]"
# print(re.findall(pattern, '.that.,'))
# '[:;=][)(]|[:][D/Pp]|[)][;:]|[(][8:]|[8][:)]|;D|<3|=D|;-[)]|:-[)(/D]|:o[)]'
# pattern = r"\w+[']\w*|\d+[./]\d+|(?!http)[\w-]+|\d+|[,.!?]"  #[^,.!?]' #((?!@\S+)\S+)'
# punc = ":;)(-PDop<38/="
pattern_email = r'^(?!\S*@)\S*$'
pattern_web = r'^(?!\S*http:)\S*$'
pattern_punc = r'^(?!\S*[.,?!])\S*$'
pattern_emoticon = r'[:;=][)(]|[:][D/Pp]|[)][;:]|[(][8:]|[8][:)]|;D|<3|=D|;-[)]|:-[)(/D]|:o[)]'
pattern = r"(:(\)|D|P|p)|(-(\)|D))|(o\)))|(;(\)|-\)|D))|(8\))|(<3)|=(\)|D)|\((:|8)"
# ' #|:(|:-)|;)|);|):|:D|(8|<3|;-)|:/|=)|:-(|8:|8)|(:|:P|;D|:o)|=(|:p|:-/|:-D|;(|=D'


# print(re.findall(pattern, ':)'))

# for each in review:
#     # pattern = r'[,.!?]|\S+'
#     # res = re.findall(pattern, each)
#     temp = each.split()
#     print(temp)
#     res = []
#     for i in temp:
#         if re.findall(pattern_email, i):
#             if re.findall(pattern_web, i):
#                 temp2 = re.findall(pattern_punc, i)
#                 if temp2:
#                     print(len(re.findall(pattern_emoticon, i))!=0)
#                     continue
#                 else:
#                     print(i)

# with open('../data/emoticons.txt', 'r', encoding='utf-8') as f:
#     emoticons = f.read().strip().split('\n')
# print(emoticons)
# for each in emoticons:
#     print(re.findall(pattern, each), end=', ')

# res = ""
# emo_lenth = len(emoticons)
# count = 0
# for each in emoticons:
#     count += 1
#     if count == emo_lenth:
#         # if each.__contains__("("|")"|"/"|r"[\]"):
#
#         res = res + re.escape(each)
#         break
#     res = res + re.escape(each) + "|"
# res = "r'" + res + "'"
# print(f'test: {res}')

# each =
# if each.__contains__("/"):
#     print('Yes')

# A = {2, 3, 5}
# B = {1, 3, 5}
#
# A = A.union(B)
# A=['i','saw']
# B={'i':1, 'saw':2}
# print(" ".join(A))
# C = "ABC"
# print(C.lower())

# ls = [['a','b','c'],['x','y','z']]
# a = random.choice(ls)
# print(a[-2:])
# print(tuple(['a', 'b']))
# print(type(str(['a', 'b'])))

ls = ['a','b','c','d']
t = tuple(ls)
# print([each for each in ls[:2]])
# t.('e')
ls.pop(0)
print(ls+['e'])
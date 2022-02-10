'''
    NUS CS4248 Assignment 1 - Objective 2 (Regular Expression, Edit Distance)

    Class EditDistanceCalculator & RegexChecker for handling Objective 2

    Important: please strictly comply with the input/output formats for
               the methods of calculate_edit_distance & approximate_matches, 
               as we will call them in testing
'''

###########################################################################
##  Suggested libraries -- uncomment the below if you want to use these  ##
##  recommended resources and libraries.                                 ##
###########################################################################

import re
import numpy as np


class EditDistanceCalculator:
    def __init__(self):
        # TODO Modify the code here
        pass

    def calculate_edit_distance(self, source, target):
        '''Calculate the edit distance from source to target

        E.g.
        [In] source="ab" target="bc"
        [Out] return 2
        '''
        # TODO Modify the code here
        if len(source) == 0:
            return len(target)
        if len(target) == 0:
            return len(source)
        if source[-1] == target[-1]:
            cost = 0
        else:
            cost = 1

        res = min(self.calculate_edit_distance(source[:-1], target) + 1,
                  self.calculate_edit_distance(source, target[:-1]) + 1,
                  self.calculate_edit_distance(source[:-1], target[:-1]) + cost)
        return res


class RegexChecker:
    def __init__(self, regex):
        '''The value of regex here should be fixed as the R_3 you've solved'''
        self.regex = regex
        # TODO Modify the code here
        self.regex = r'(?=(^(\S)[\S]*\2)$)(?=(^(?!.*(\S)(\S)\5\4).*$))'
        self.regex_abba = r'((\S)(\S)\3\2)'

    def matches(self, word):
        '''Return whether a word is (exactly) matched by the regex'''
        # TODO Modify the code here
        temp = re.findall(self.regex, word)
        if len(temp) == 0:
            return False
        else:
            return True

    def matches_abba(self, word):
        temp = re.findall(self.regex_abba, word)
        return len(temp)

    def approximate_matches(self, word, k=2):
        '''Return whether a word can be matched by the regex within k 
        errors (edit distance)
        You can assume that the word is got from a corpus which has 
        already been tokenized.

        E.g.
        [In] word="blabla"
        [Out] return True
        '''
        # TODO Modify the code here
        cost = 0

        if self.matches(word):
            if cost <= k:
                return True
            else:
                return False

        if word[0] == word[-1]:
            cost += self.matches_abba(word)
        elif word[0] != word[-1]:
            temp1 = word.replace(word[0], word[-1], 1)
            temp2 = word.replace(word[-1], word[0], 1)
            cost1 = 1 + self.matches_abba(temp1)
            cost2 = 1 + self.matches_abba(temp2)
            cost = min(cost1, cost2)

        if cost <= k:
            return True
        else:
            return False


# res = RegexChecker('sdf')
# print(res.approximate_matches('abba', k=1))
# print(res.calculate_edit_distance('Sunday', 'Saturday'))

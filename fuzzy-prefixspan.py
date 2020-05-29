import sys
import math
import numpy as np
import pandas as pd
from time import time
from itertools import product
from datetime import datetime 

SUP_MIN = 0.05
GENERAL_ERROR = 0
INPUT_VAL_ERROR = 1
INDEX_OUT_ERROR = 2
DATA_TYPE_ERROR = 3
TERM_SET = ['short', 'middle', 'long']

def t_mal(points, point, scope=[0,30]):
    """Fuzzy value calculation function
    
    Receive four points as functions parameters and input value to 
    calculate fuzzy value.
    
    Args:
        points: four fuzzy function parameters
            trapezoidal: four different values
            triangular: 2nd and 3rd the same value
        point: input value to calculate membership value
        scope: scope of valid value
        
    Returns:
        return fuzzy value of membership function [0, 1]
    Raises:
        INPUT_VAL_ERROR: point not in scope
    """
    if point < 0.0:
        return 0.0
    if point < scope[0] or point > scope[1]:
        return 0.0
        # unsolved yet!
        # sys.exit(INPUT_VAL_ERROR)
    elif point < points[0] or point > points[3]: return 0.0
    elif point >= points[1] and point <= points[2]: return 1.0
    elif point < points[1]: return (point-points[0])/(points[1]-points[0])
    elif point > points[2]: return (points[3]-point)/(points[3]-points[2])


def cal_interval(term, val):
    """Given term, value return membership of this value to this term
    
    Given term, value return membership of this value to this term, 
    membership function parameters set inside, this could be changed 
    to configurable in the future.
    
    Args:
        term: time interval term, currently support: short, middle, long
        val: value of time interval, integer type
    
    Returns:
        return fuzzy membership value of given term and interval value
    """
    term_set = ['short', 'middle', 'long']
    if term not in term_set: sys.exit(INPUT_VAL_ERROR)
    
    SHORT_0, SHORT_1, SHORT_2, SHORT_3 = 0, 0, 3, 7
    MIDDLE_0, MIDDLE_1, MIDDLE_2, MIDDLE_3 = 6, 10, 15, 20
    LONG_0, LONG_1, LONG_2, LONG_3 = 14, 25, 30, 30
    
    if term == 'short': return t_mal([SHORT_0, SHORT_1, SHORT_2, SHORT_3], val)
    elif term == 'middle': return t_mal([MIDDLE_0, MIDDLE_1, MIDDLE_2, MIDDLE_3], val)
    elif term == 'long': return t_mal([LONG_0, LONG_1, LONG_2, LONG_3], val)
    
def _get_freq_items(data, threshold=SUP_MIN):
    """Return frequent items from data(column 1) given threshold
    
    Args:
        data: pandas DataFrame table with 3 columns: sid, item, date
        
    Returns:
        return frequent item list
    """
    set_dict_1 = {}
    pattern_0 = []
    for i in range(data.shape[0]):
        if data.iloc[i,1] in set_dict_1: set_dict_1[data.iloc[i,1]].add(data.iloc[i,0])
        else: set_dict_1[data.iloc[i,1]] = set([data.iloc[i,0]])
    for item in set_dict_1:
        set_dict_1[item] = len(set_dict_1[item])/len(np.unique(data.iloc[:,0].to_list()))
        if set_dict_1[item] >= threshold: 
            pattern_0.append(item)
    return pattern_0

def hashing(pattern):
    """hash value of list of all string items"""
    if type(pattern)==str: return pattern
    tmp_items = []
    for item in pattern: tmp_items.append(str(item))
    try:
        string = ''.join(tmp_items)
    except:
        print(tmp_items)
        sys.exit(DATA_TYPE_ERROR)
    return hash(string)

def match_phase1(k, prefix, postfix, frequent_set, sid_l=10, term_set=TERM_SET, sup_min=SUP_MIN):
    """match phase1 process of ProfixSpan algorithm
    First phase: it takes data table (k==1) or postfix (k>1) and return possible patterns
    
    Args:
        k: 1 for data table 2 for postfix as input data
        prefix: 1 of k as frequent itemset, 2 for prefix patterns
        postfix: projected postfix data table
        frequent_set: frequent itemset (length==1)
        sid_l: all unique records according to sid
        term_set: all supported time interval terms
        sup_min: minimum support value threshold
        
    Returns:
        return phase1 result:
        { 'a': [sid, start_time, postfix, prefix_support], ...
        
        } or 
        { #hash value of long pattern: [sid, start_time, postfix, prefix_support, *prefix*], ...
        }
    """
    
    def _construct_result(result, prefix, data):
        '''constructing result dictionary'''
        key = hashing(prefix)
        if key in result:
            result[key].append(data)
        else:
            result[key] = [data]
        pass
    
    if k==1: # prefix is data
        result = {}
        this_sid = 0
        counter = 0
        for i in range(postfix.shape[0]):
            if postfix.iloc[i,0] != this_sid:
                this_sid = postfix.iloc[i,0]
                counter = 0
            else: counter += 1
            if postfix.iloc[i,1] in prefix: # if freq item
                pre_sliced = postfix[postfix['id'] == 
                                 postfix.iloc[i,0]].iloc[:counter+1,1:] # slice next items
                pre_sliced = pre_sliced[pre_sliced.apply(lambda x: x['data'] in 
                                             frequent_set, axis=1)] # in same sid block
                sliced = postfix[postfix['id'] == 
                                 postfix.iloc[i,0]].iloc[counter+1:,1:] # slice next items
                sliced = sliced[sliced.apply(lambda x: x['data'] in 
                                             frequent_set, axis=1)] # in same sid block
                if sliced.shape[0] > 0: # has more for further
                    
                    for k in range(sliced.shape[0]):
                        if (sliced.iloc[k,0] != postfix.iloc[i,1] and
                            sliced.iloc[k,0] in prefix and
                            sliced.iloc[k,1] != postfix.iloc[i,2]):
                            splict_point = k
                    
                            post_fix = [[sliced.iloc[j,0], sliced.iloc[j,1]] 
                                        for j in range(splict_point, sliced.shape[0])]
                            skip_fix = [[sliced.iloc[j,0], sliced.iloc[j,1]] 
                                        for j in range(splict_point)]
                            pre_fix = [[pre_sliced.iloc[j,0], pre_sliced.iloc[j,1]] 
                                        for j in range(pre_sliced.shape[0])] + skip_fix
                            sid = postfix.iloc[i,0]
                            start_time = postfix.iloc[i,2]
                            prefix_support = 1 # default sup for single item
                            _construct_result(result, postfix.iloc[i,1], [sid, start_time, 
                                                                  post_fix, prefix_support, pre_fix])
    elif k==2: # prefix is post table
        # postfix = hashes:[map_hash_pat, map_hash_post, map_hash_support]
        result = {}
        for pat in prefix:
            post_block = postfix[hashing(pat)] # post_block -> [ , [[(),()], [(),()]], ] ?
            support_prior = post_block[2]
            posts = post_block[1] # len:5
            for post in posts: # loop posts
                if len(post[2]) > 0:
                    sid = post[0]
                    start_time = post[1]
                    counter = 0
                    #{prefix: [[sid, start_time, postfix, prefix_support], ...]}
                    for pair in post[2]: # [(item, date) ...]
                        if pair[1] != start_time:
                            if pair[0] in frequent_set and pair[0] not in pat:
                                _construct_result(result, pat, 
                                                  [sid, start_time, post[2][counter:], 
                                                   post[3], post[4]])
                        counter += 1
    return result

def match_phase2(k, phase1, frequent_set, frequent_pats, sid_l=10, term_set=TERM_SET, sup_min=SUP_MIN):
    """match phase2 process of ProfixSpan algorithm
    Second phase: it takes new found patterns with sid, support, post etc, diction type
    key: item(k==1) or patter hash value (k>1) and merged patterns(for same one, merging 
    support for each sid)
    
    Args:
        k: 1 for item as key 2 for long pattern hash value as key
        phase1: pattern diction need to evaluate average support
        frequent_set: frequent itemset (length==1)
        sid_l: all unique records according to sid
        term_set: all supported time interval terms
        sup_min: minimum support value threshold
        
    Returns:
        return frequent pattern list and frequent pattern's support and post
        fre_pat: [[pattern], [pattern] ...]
        map_hash_combined: {prefix: [[term, pattern, support]]}
        $pattern: [sid, post[i][1], post[i+1:], support, pre]
    """
    # phase1: {prefix: [sid, start_time, postfix, prefix_support, *prefix*]}
    # *return: {prefix: [[term, pattern, support]]}
    def _hash2pat(pat, freq_set=frequent_pats, k=k):
        """use hash value to get pattern"""
        if k == 1: return pat
        for item in freq_set:
            if pat == hashing(item): return item
        print('error', pat, freq_set)
    
    def _cal_interval(timestamp_end, timestamp_start):
        """calculate interval of two timestamps"""
        if type(post[i][1]) == str:
            return (datetime.strptime(timestamp_end, '%Y-%m-%d') - \
                        datetime.strptime(timestamp_start, '%Y-%m-%d')).days 
        elif type(post[i][1]) == np.int64:
            return timestamp_end - timestamp_start
        else:
            sys.exit(DATA_TYPE_ERROR)
    
    def _append_new_pattern(pattern, term, item):
        """append new term and item to pattern"""
        if type(_hash2pat(pattern)) == list:
            return _hash2pat(pattern) + [term, item]
        else: return [_hash2pat(pattern)] + [term, item]
        
    def _update_hash_dict(dict_pat, dict_merged, dict_post,
                          tmp_pat, sid, support, post, pre, i):
        """update new pattern into dictionaries"""
        if hashing(tmp_pat) not in map_hash_pat:
            dict_pat[hashing(tmp_pat)] = tmp_pat
            dict_merged[hashing(tmp_pat)] = dict({sid:[support]})
            dict_post[hashing(tmp_pat)] = [[sid, post[i][1], post[i+1:], support, pre+post[:i+1]]]
        else:
            # line[0]: sid, hashing(line[2]): hash of pattern, line[3]: support
            if sid in dict_merged[hashing(tmp_pat)]: # sid to add 
                # same sid expand
                dict_merged[hashing(tmp_pat)][sid].append(support)
            else: 
                dict_merged[hashing(tmp_pat)][sid] = [support]
            dict_post[hashing(tmp_pat)].append([sid, post[i][1], post[i+1:], support, pre+post[:i+1]])
            
    def _max_pre_support(prelist, prepat, term_set=term_set):
        """calculate maximum prefix support if multiple matches"""
        raw_pat = [item for item in prepat if item not in term_set]
        raw_term = [item for item in prepat if item in term_set]
        raw_prelist = [pair[0] for pair in prelist]
        dict_pat = []
        max_support = 0
        for item in raw_pat:
            matches = [prelist[i][1] for i in range(len(raw_prelist)) if raw_prelist[i] == item]
            if len(matches) > 0:
                dict_pat.append(matches)
            else:
                dict_pat.append([-1])
        combinations = list(product(*dict_pat))
        supports = []
        divid = (len(prepat)-1)/2
        for indexes in combinations:
            interval = [_cal_interval(indexes[i],indexes[i-1]) if (indexes[i]!=-1 and indexes[i-1]!=-1) else -1 for i in range(1, len(indexes))]
            support = 0
            for i in range(len(interval)):
                one_support = cal_interval(raw_term[i], interval[i])
                if one_support == 1: return 1
                else: support += one_support
            supports.append(support/divid)
        return max(supports)
    map_hash_pat, map_hash_merged, map_hash_post, map_hash_pre = {}, {}, {}, {}
    # [sid, term, pat, pat's sup, start time, post]
    counter = 0
    for prefix in phase1:
        for sequence in phase1[prefix]:
            post = sequence[2]
            pre_fix = sequence[4]
            item = post[0][0]
            i = 0
            interval = _cal_interval(post[i][1], sequence[1])
            sid = sequence[0]
            support_prior = sequence[3]
            # _max_pre_support(prelist, prepat, term_set=term_set)
            the_term = ""
            support = 0
            for term in term_set:
                counter += 1
                # NOTICE: step with one item and min prior sup and new sup
                tmp_support = cal_interval(term, interval)
                if tmp_support > support:
                    support = tmp_support
                    the_term = term
            if support <= sup_min:
                continue
            if type(prefix) == str:
                test_support = sequence[3]
                l = 2
            else: 
                pat = _hash2pat(prefix)
                test_support = _max_pre_support(pre_fix, pat)
                l = (len(pat)+1)/2
            support = support*(l-1)/l + test_support/l
            if support > sup_min:
                tmp_pat = _append_new_pattern(prefix, the_term, item)
                # find suitable pattern write it in the dictionary
                _update_hash_dict(map_hash_pat, map_hash_merged, map_hash_post,
                                 tmp_pat, sid, support, post, pre_fix, i)
    
    print('counter:', counter)
    # filtering with new support value
    map_hash_support = {hashes:0 for hashes in map_hash_merged}
    for hash_val in map_hash_merged: # compute average support
        map_hash_support[hash_val] = sum([max(map_hash_merged[hash_val][sid]) 
                                          for sid in map_hash_merged[hash_val]])/sid_l
    
    fre_pat = [map_hash_pat[hashes] for hashes in map_hash_pat if map_hash_support[hashes] > SUP_MIN]
    # merge three dictionary (hash: pat, post(with sup), pat's sup)
    map_hash_combined = { hashes:[map_hash_pat[hashes], map_hash_post[hashes], map_hash_support[hashes]]
                        for hashes in map_hash_support if map_hash_pat[hashes] in fre_pat }
    return fre_pat, map_hash_combined
    

def test(data, item_threshold=0.2):
    '''main function for testing'''
    def _if_terminate(map_hash_combined):
        for pat in map_hash_combined:
            for pair in map_hash_combined[pat][1]: 
                if len(pair[2])>0: return False
        else: return True
    def _display_patterns(fre_pat, pat_info, i=3):
        '''display all patterns or part of if too long'''
        l = min(len(fre_pat), i)
        etc = '... {} more'.format(len(fre_pat)-l) if len(fre_pat)-l > 0 else ''
        print('Pat:', fre_pat[:l], etc)
        supports = [pat_info[prefix][-1] for prefix in pat_info]
        print('Sup:', supports[:l], etc, '<max:{}>'.format(max(supports)))
    
    pattern_0 = get_freq_items(data, item_threshold)
    print('frequent items:', pattern_0)
    sid_l = len(np.unique(data['id']))
    print('sid length:', sid_l)
    phase1 = match_phase1(1, pattern_0, data, pattern_0)
    #print(phase1)
    fre_pat, map_hash_combined = match_phase2(1, phase1, pattern_0, pattern_0, sid_l=sid_l)
    time_blocks = []
    print('initial pattern:')
    _display_patterns(fre_pat, map_hash_combined)
    while not _if_terminate(map_hash_combined):
        print('***')
        phase1 = match_phase1(2, fre_pat, map_hash_combined, pattern_0)
        fre_pat, map_hash_combined = match_phase2(2, phase1, pattern_0, fre_pat, sid_l=sid_l)
        if (len(fre_pat)>0):
            _display_patterns(fre_pat, map_hash_combined)
        #print(map_hash_combined)
    #print(map_hash_combined)

table = pd.read_csv('./data.csv') # data input
table.columns = ['id','data','date']

SUP_MIN = 0.05
ITEM_MIN = 0.1
t = time()
test(table.iloc[:1000,:], ITEM_MIN)
print("final time:", time()-t)
import re
from konlpy.tag import Kkma
from collections import Counter
from .eda.analysis_duplicates import word_counter
from .eda.analysis_prefix import find_common_prefix
from .eda.analysis_substr import find_common_substr
from .eda.split_sentences import split_sentences
from .find_lrs import find_longest_repeated_substring
import os
import sys

def check_duplication(desc, attrs=[]):
    desc = desc.replace("'", "")
    desc = desc.replace("반소매", "반팔 소매")
    desc = desc.replace("긴소매", "긴팔 소매")
    attrs = ','.join(attrs)
    attrs = attrs.replace("반소매", "반팔")
    attrs = attrs.replace("긴소매", "긴팔")
    sentences = split_sentences(desc)
    prefix = find_common_prefix(sentences)
    if len(prefix) >= 2:
        return f"common prefix: {prefix}"
    substr = find_common_substr(sentences)
    if len(substr) >= 8:
        return f"common substr: {substr}"
    lrss = [find_longest_repeated_substring(x) for x in sentences]
    lrss = [x for x in lrss if len(x) >= 5]
    if lrss:
        return f"longest repeated substr: " + ','.join(lrss)

    counters = [word_counter(y) for y in sentences]

    # rep: input attrs
    all_dups = sum(counters, start=Counter()).most_common()
    all_dups = [a for a, b in all_dups if b > 1]
    input_dups = [x for x in all_dups if x[0] in attrs]
    if input_dups:
        return "input duplicates: " + str(input_dups)

    # rep: intra-sentence nouns
    intra = [z for y in counters for z in y.most_common(1)]
    intra = [a for a, b in intra if b > 1]
    if intra:
        return "intra sentence duplicates: " + str(intra)
    if any(x.count('핏') > 1 for x in sentences):
        return "intra sentence duplicates: 핏"

    # rep: inter-sentecne long-length nouns
    inter = sum(counters, start=Counter()).most_common()
    inter = [a for a, b in inter if len(a[0]) >= 3 and b >= 2]
    if inter:
        return "inter sentence duplicates: " + str(inter)

if __name__ == "__main__":
    model = sys.argv[1]
    gold_or_sample = sys.argv[2]
    post_fix = sys.argv[3]
    

    attrs = []
    with open("/home/hzlcodus/codes/ssf-metrics/attr_ko_dict.txt", "r") as f:
        attrlines = f.readlines()
        for line in attrlines:
            line = line.split(":")[1].split(", ")
            for i in range(len(line)):
                line[i] = line[i].strip()
            attrs.extend(line)
    
    with open(f"/home/hzlcodus/codes/peft/outputs/{model}_test_extract_{gold_or_sample}{post_fix}", "r") as f:
        desclines = f.readlines()
        with open("/home/hzlcodus/codes/ssf-metrics/duplication_sample.txt", "w") as wf:
            for desc in desclines:
                desc = desc.strip()
                print(desc, file=wf)
                print(check_duplication(desc, attrs), file=wf)
                print("====================================", file=wf)


    with open("/home/hzlcodus/codes/ssf-metrics/duplication_sample.txt", "r") as f:
        lines = f.readlines()
        line_num = 0
        dup_num = 0
        for index, line in enumerate(lines):
            if line.startswith("="):
                line_num += 1
                if not lines[index-1].strip().startswith("None"):
                    dup_num += 1
        print (f"line_num: {line_num}, dup_num: {dup_num}, duplication rate: {dup_num/line_num*100}%")

    with open(f"/home/hzlcodus/codes/peft/outputs/{model}_test_ace_{gold_or_sample}{post_fix}", "r") as f:
        desclines = f.readlines()
        with open("/home/hzlcodus/codes/ssf-metrics/duplication_gold.txt", "w") as wf:
            for desc in desclines:
                if desc.strip() == "":
                    continue
                desc = desc.strip()
                print(desc, file=wf)
                print(check_duplication(desc, attrs), file=wf)
                print("====================================", file=wf)
    
    with open("/home/hzlcodus/codes/ssf-metrics/duplication_gold.txt", "r") as f:
        lines = f.readlines()
        line_num = 0
        dup_num = 0
        for index, line in enumerate(lines):
            if line.startswith("="):
                line_num += 1
                if not lines[index-1].strip().startswith("None"):
                    dup_num += 1
        print (f"line_num: {line_num}, dup_num: {dup_num}, duplication rate: {dup_num/line_num*100}%")
from konlpy.tag import Okt
import os
import sys
tagger = Okt()


model = sys.argv[1]
gold_or_sample = sys.argv[2]
post_fix = sys.argv[3]

pos_set = set()
with open(f"/home/hzlcodus/codes/peft/outputs/{model}_test_extract_{gold_or_sample}{post_fix}", "r") as f:
    poslines = f.readlines()
    for line in poslines:
        pos = tagger.pos(line)
        for p in pos:
            pos_set.add(p)

print("pos set num", len(pos_set))

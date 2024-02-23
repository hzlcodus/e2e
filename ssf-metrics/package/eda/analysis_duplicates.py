from konlpy.tag import Kkma
from collections import Counter

Kkma()


def word_counter(txt):
    results = Kkma().pos(txt)
    results = [x for x in results if len(x[0]) > 1]
    results = [x for x in results if x[1] != 'NR'] 
    results = [x for x in results if x[1].startswith('N') or x[1] in ['XR', 'UN']]
    return Counter(results) # Counter({('소매', 'NNG'): 2, ('반팔', 'NNG'): 1, ('긴팔', 'NNG'): 1})


# if __name__ == "__main__":
#     from preprocess import read_data
#     from split_sentences import split_sentences
#     from tqdm import tqdm
#     data = read_data()
#     intra_counter = Counter()
#     inter_counter = Counter()
#     for x in tqdm(data.desc):
#         sentences = split_sentences(x)
#         counters = [word_counter(y) for y in sentences]
#         intra_counter[max(z for y in counters for z in y.values())] += 1
#         inter_counter[sum(counters, start=Counter()).most_common(1)[0][1]] += 1
#     print('intra:', intra_counter)
#     print('inter:', inter_counter)

# 1: 5004   # 84.8%
# 2: 881    # 99.8%
# 3: 12
# 4: 1

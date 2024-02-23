import os
from itertools import combinations


def find_common_prefix(sentences):
    prefixes = [os.path.commonprefix(x) for x in combinations(sentences, 2)]
    # combination은 말 그대로 xC2를 구하는 것이므로, 2개씩 pair로 묶어서 commonprefix를 구하고,
    # 그 중 가장 긴 것을 찾아서 반환한다.
    if prefixes:
        return max(prefixes, key=lambda x: len(x))
    else:
        return ''


# if __name__ == "__main__":
#     from preprocess import read_data
#     from split_sentences import split_sentences
#     from collections import Counter
#     data = read_data()
#     counter = Counter()
#     for x in data.desc:
#         sentences = split_sentences(x)
#         prefix = find_common_prefix(sentences)
#         counter[len(prefix)] += 1
#     print(counter)

# 0: 5745   → 97.41%, 아예 첫글자부터 다르게 해도 되기는 함
# 1: 87     → 98.88%
# 2: 21     → 99.24%, (1.33 글자 이상 밴) = (2 글자 이상 밴)
# 3: 8
# 4: 17
# 5: 7
# 6: 5
# 7: 4
# 9: 2
# 11: 2

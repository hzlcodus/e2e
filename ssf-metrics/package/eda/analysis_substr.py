from difflib import SequenceMatcher
from itertools import combinations


def commonsubstr(string1, string2): # 공통 substring을 찾는 함수
    match = SequenceMatcher(None, string1, string2).find_longest_match()
    return string1[match.a:match.a + match.size]


def find_common_substr(sentences): 
    substrs = [commonsubstr(*x) for x in combinations(sentences, 2)]
    if not substrs:
        print("sentences:", sentences)
        return ''
    return max(substrs, key=lambda x: len(x)) # 모든 sentence 쌍 조합 중 가장 긴 substring을 찾아서 반환


# if __name__ == "__main__":
#     from preprocess import read_data
#     from split_sentences import split_sentences
#     from collections import Counter
#     data = read_data()
#     counter = Counter()
#     for x in data.desc:
#         sentences = split_sentences(x)
#         substr = find_common_substr(sentences)
#         counter[len(substr)] += 1
#     print(counter)

# 3: 1571   → 26.6%
# 4: 3045   → 78.3%
# 5: 545    → 87.5%
# 6: 243    → 91.6%
# 7: 126    → 93.8%
# 8: 203    → 97.2%
# 9: 138    → 99.5%, 9글자 이상이면 밴
# 10: 7
# 11: 11
# 12: 8
# 13: 1

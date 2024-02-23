def build_suffix_array(s):
    '''
    input: s (str)
    output: suffix array (list of int), with each element representing the starting index of the suffix, sorted in lexicographical order
    '''
    n = len(s)
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    #print("suffixes:", suffixes)
    return [index for _, index in suffixes] # 띄어쓰기 안 빼도 되나?


def build_lcp_array(s, suffix_array):
    n = len(s)
    rank = [0] * n
    for i in range(n):
        rank[suffix_array[i]] = i
    lcp = [0] * n # lcp is the longest common prefix between the i-th and (i+1)-th suffixes in the suffix array
    k = 0 # k is the length of the longest common prefix between the i-th and (i+1)-th suffixes
    for i in range(n):
        if rank[i] == n - 1:
            k = 0
            continue
        j = suffix_array[rank[i] + 1] # j is the index of the suffix that comes after the i-th suffix in the suffix array
        #print("i:", i, "j:", j)
        while i + k < n and j + k < n and s[i + k] == s[j + k]: # compare the characters of the i-th and j-th suffixes
            #print("s[i+k]:", s[i+k])
            k += 1
        lcp[rank[i]] = k
        if k > 0:
            k -= 1 # decrease k because the next suffixes will have at least k - 1 characters in common
    return lcp


def find_longest_repeated_substring(s):
    suffix_array = build_suffix_array(s)
    #print("suffix_array:", suffix_array)
    lcp_array = build_lcp_array(s, suffix_array)
    max_length = max(lcp_array)
    index = lcp_array.index(max_length)
    return s[suffix_array[index]:suffix_array[index] + max_length]


if __name__ == "__main__": 
    test_string = "라운드넥이 여성적인 매력을 더하며 반소매 소매가 시원한 느낌을 줍니다."
    #test_string = "하이넥으로 목까지 감싸주어 찬바람을 차단해주며, 나일론 소재로 가벼운 착용감과 통기성이 뛰어나 봄부터 가을까지 활용도가 높습니다."
    print(find_longest_repeated_substring(test_string))

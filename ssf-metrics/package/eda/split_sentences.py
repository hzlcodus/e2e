import re


def split_sentences(d):
    # phase 1: 온점 기준으로 나누기
    # phase 2: 문장이 '다'/'요'/'닏'으로 끝나는지 확인
    d = d.strip()
    d += '' if d.endswith('.') else '.'
    p1 = re.split("(\.+)", d) # "Hello...World!..Goodbye." → ['Hello', '...', 'World!', '..', 'Goodbye', '.']
    p1 = list(zip(p1[0::2], p1[1::2])) # [('Hello', '...'), ('World!', '..'), ('Goodbye', '.')]
    p2 = [''.join(p1.pop())] # ['Goodbye.']
    for x, y in p1[::-1]: # ('World!', '..') → ('Hello', '...')
        if x[-1] in {'다', '요', '닏'}:
            p2.append(x + y) # append 'World!..'
        else:
            p2.append(x + y + ''.join(p2.pop())) # append 'World!..Goodbye.'
    return [x.strip() for x in p2[::-1]] # do [::-1] to reverse the list #['Hello', 'World!', 'Goodbye.']


# if __name__ == "__main__":
#     from preprocess import read_data
#     data = read_data()
#     for x in data.desc:
#         for y in split_sentences(x):
#             if '.' in y[:-1]:
#                 print(y)

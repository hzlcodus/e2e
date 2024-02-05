# given a data like this: 
'''
name : 립배색 그래픽 반소매 티셔츠 - 라이트그레이 | season : FW23 | brand : 8seconds | color : 라이트그레이 | sleeves_length : 반소매 | style : 티셔츠 | material : 데님, 코튼 | pattern/print : 레터링
립 배색과 깔끔한 레터링 그래픽이 포인트로 적용된 반소매 티셔츠입니다. 톡톡한 두께감의 코튼 소재가 사용되었으며, 배색 포인트와 그래픽이 캐주얼한 느낌을 주어 데님이나 캐주얼 팬츠와 코디하기 좋은 아이템입니다.

'''
# the attributes are in attr.txt
# the descriptions are in desc.txt, with same line number as attr.txt

# for each description, tell the percentage of attributes included in the description, except for names
# if more than two attributes for on category (divided by |), divide by comma

import sys
model = sys.argv[1]

# make a dictionary of attributes' synonyms
syn_dict = {}
with open('/data/hzlcodus/synonyms.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        key, value = line.split(': ')
        key = key.strip()
        value = value.strip()
        #make value to list
        value = value.replace('[', '').replace(']', '').replace('\'', '').split(', ')
        if value[0] == '':
            value = []
        syn_dict[key] = value

def find_attr_in_desc(attr, desc):
    if attr in desc:
        return True
    for syn in syn_dict[attr]:
        if syn in desc:
            return True
    return False
    
def main(gold_or_sample):

    with open(f'/home/hzlcodus/codes/peft/outputs/{model}_test_extract_src', 'r') as f:
        attr_lines = f.readlines()
    with open(f'/home/hzlcodus/codes/peft/outputs/{model}_test_extract_{gold_or_sample}', 'r') as f:
        desc_lines = f.readlines()

    with open('/data/hzlcodus/percentage.txt', 'w') as f:
        line_count = 0
        line_sum = 0
        for i in range(len(attr_lines)):
            attr = attr_lines[i].replace('<|endoftext|>', '').strip().split(' | ')
            if gold_or_sample == 'gold':
                desc = desc_lines[2*i].strip()
            else:
                desc = desc_lines[i].strip()
            attr_count = 0
            total_attr_count = 0
            #print (attr, desc)
            for a in attr:
                if a.split(' : ')[0] != 'name' and a.split(' : ')[0] != 'season' and a.split(' : ')[0] != 'brand' and a.split(' : ')[0] != 'color':
                    a = a.split(' : ')[1].strip()
                    #print("a: ", a)
                    if ',' in a:
                        a = a.split(', ')
                        for aa in a:
                            total_attr_count += 1
                            if find_attr_in_desc(aa, desc):
                                attr_count += 1
                    else:
                        total_attr_count += 1
                        if find_attr_in_desc(a, desc):
                            attr_count += 1
                            #print(a)

            if total_attr_count == 0:
                print('None', file=f)
            else:
                print(attr_count/total_attr_count, file=f)
                line_count += 1
                line_sum += attr_count/total_attr_count


    print(gold_or_sample, "Recall :", line_sum/line_count)

if __name__ == "__main__":
    main(gold_or_sample='gold')
    main(gold_or_sample='sample')


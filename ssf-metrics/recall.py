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
style_attrs = ['플레어 원피스', '립 스웨터', '볼레로 카디건', '니트패치 셔츠', '밴딩 팬츠', '패딩 코트', '부츠컷 팬츠', '팬츠', '베스트 점퍼', '트렌치 코트', '더플 코트', '트랙재킷', '벨티드 팬츠', '카고 스커트', '스포츠 팬츠', '코듀로이 스커트', '라이더재킷', '퍼 재킷', '후드 티셔츠', '레더 코트', '서스펜더 스커트', '자파리 아우터', '후드 롱 아우터', '심리스 푸퍼', '바이커 스커트', '케이블 카디건', '다운 재킷', '티어드 원피스', '퍼 아우터', '퍼 베스트', '랩 원피스', '후드 카디건', '하이브리드재킷', '조거팬츠', '다운 코트', '캐주얼 팬츠', '하이브리드 셔츠', '무스탕', '데님 원피스', '데님 스커트', '수트 베스트', '시어링 재킷', '테니스 스커트', '트랙팬츠', '코듀로이 팬츠',
'스웨터 팬츠', '랩스커트', '숏 푸퍼재킷', '포멀 팬츠', '치노팬츠', '셔츠 원피스', '하프터틀넥 스웨터', '숏 점퍼', '스웻팬츠', '케이프 코트', '카 코트', '패러슈트 팬츠', '원피스', '후드 스웨터', '랩 블라우스', '포멀재킷', '셔츠 블라우스', '패딩셔츠', '티어드 스커트', '스커트', '블라우스', '아우터재킷', '바이커 팬츠', '포멀 원피스', '무스탕 코트', '럭비 스웨터', '베스트 스웨터', '로브 코트', '밴딩 스커트', '가죽 재킷', '심리스 재킷', '코르셋', '티셔츠', '맥 코트', '뷔스티에 원피스', '크리켓 스웨터', '레인코트', '주름 원피스', '니트 원피스', '칼라 스웨터', '앙고라 스웨터', '랩 스커트', '스웨터 베스트', 'BDU 재킷', '하이웨스트 팬츠', '케이프 카디건', '밀리터리 패딩',
'주름 팬츠', '주름 스커트', '오버롤 팬츠', '캐주얼 원피스', '저지 원피스', '패딩 베스트', '수트 재킷', '패딩 재킷', '셋인 코트', '다운 베스트', '롱패딩', '인타샤 스웨터', '데님 셔츠', '플레인 카디건', '구르카 팬츠', '레인 코트', '후드 푸퍼', '마운틴 파카', '플레인 스웨터', '다운 패딩', '사파리 재킷', '긴소매 티셔츠', '브이넥 스웨터', '레이어드 스커트', '셔츠', '헤링턴 재킷', '스트링 팬츠', '카고팬츠', '헤비 파카', '하프밴딩 팬츠', '투피스 수트', '하프슬리브 스웨터', '코트', '터틀넥 스웨터', '집업 스웨터', '핸드메이드 코트', '폴로셔츠', '퍼 코트', '서커 팬츠', '우븐 탑', '스타디움 재킷', '초어재킷', '데님셔츠', '데님 팬츠', '점퍼', '다운 점퍼', '데님 탑', '비치 원피스', 
'드리즐러 재킷', '벨티드 스커트', '패딩 셔츠', '하이웨이스트 스커트', '트러커재킷', '플레어 스커트', '헌팅 재킷', '후드 숏 아우터', '발마칸 코트', '코트 카디건', '스웨터 원피스', '탑', '3피스 수트', '패디드 점퍼', '케이블 스웨터', '레깅스', '카디건', '체스터 코트', '오버롤 원피스', '데님 재킷', '아노락', '퀼팅 재킷', '캐주얼셔츠', '버뮤다팬츠', '셔츠 재킷', '반소매 티셔츠', 'MA-1 스커트', '베스트', '스웨트셔츠', '럭비셔츠', '슬랙스', '라운드 스웨터', '스웨터 스커트', '하이브리드 팬츠', '캐주얼 코트', '뷔스티에', '스커트 팬츠', '아웃도어 베스트', '집업 카디건', '재킷', '슬릿 스커트', '점프수트', '스웨터', '블레이저', '크롭탑', '필드재킷', '바시티재킷', '아우터형 셔츠',
 '스웨터 머플러', '밀리터리 코트', '윈드브레이커', '드레스셔츠', '스카쟌 재킷', '경량 패딩', '샤켓재킷', '클래식 원피스', '리버시블 점퍼', '로브 카디건', '블루종', '시폰 블라우스']


# make a dictionary of attributes' synonyms
syn_dict = {}
with open('/home/hzlcodus/data/synonyms.txt', 'r') as f:
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
    if attr in style_attrs:
        desc = desc.split('.')[0]
    if ' '+attr in desc:
        return True
    return False

    
def main(gold_or_sample):

    with open(f'/home/hzlcodus/codes/peft/outputs/{model}_test_ace_src', 'r') as f:
        attr_lines = f.readlines()
    with open(f'/home/hzlcodus/codes/peft/outputs/{model}_test_ace_{gold_or_sample}', 'r') as f:
        desc_lines = f.readlines()

    with open('/data/hzlcodus/percentage.txt', 'w') as f:
        line_count = 0
        line_sum = 0
        for i in range(len(attr_lines)):
            attr_line = attr_lines[i].replace('<|endoftext|>', '').strip()
            attr = attr_line.split(' | ')
            if gold_or_sample == 'gold':
                desc = desc_lines[2*i].strip()
            else:
                desc = desc_lines[i].strip()
            attr_count = 0
            total_attr_count = 0
            desc = ' '+desc # 앞에 space 있는 상태로 검색하기 때문에
            #print (attr, desc)
            for a in attr:
                if a.split(' : ')[0] != 'name' and a.split(' : ')[0] != 'season' and a.split(' : ')[0] != 'brand' and a.split(' : ')[0] != 'color' and a.split(' : ')[0] != 'sub-brand':
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
                        else:
                            #print(a, "/", attr_line, "/", desc)
                            pass
            if total_attr_count == 0:
                print('None', file=f)
            else:
                print(attr_count/total_attr_count, file=f)
                #print("attr_count: ", attr_count)
                line_count += 1
                line_sum += attr_count/total_attr_count


    print(gold_or_sample, "Recall :", line_sum/line_count)

if __name__ == "__main__":
    main(gold_or_sample='gold')
    main(gold_or_sample='sample_5')


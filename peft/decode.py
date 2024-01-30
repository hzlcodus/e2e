import argparse
import logging

import numpy as np
import torch
import json
import sys


from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig,
)
import sys, os
from peft import PeftConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def read_e2e_files(path, tokenizer, lowdata_token=None):
    file_dict = {}
    with open(path, 'r') as f:
        for line in f:
            src, tgt = line.strip().split('||')
            # URGENT CHANGE
            # src =  src + ' {}'.format(' summarize :')
            if lowdata_token is None:
                src = ' {} {}'.format(src, tokenizer.bos_token)
                # src =  src + ' {}'.format(tokenizer.bos_token)
            else:
                src = ' {} {} {}'.format(lowdata_token, src, tokenizer.bos_token)
            if src not in file_dict:
                file_dict[src] = []
            file_dict[src].append(tgt)
    return file_dict


def write_e2e_corr(prompt_lst, file_dict, corr_path):
    print(len(prompt_lst))
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            for line in file_dict[x]:
                if not line.strip():
                    print('PROBLEM', line,'PROBLEM',file_dict[x] )
                else:
                    print(line, file=f)
            print('', file=f)

def write_e2e_src(prompt_lst, corr_path):
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            print(x, file=f)
    return


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = 10000  # avoid infinite loop. jcy : MAX_LENGTH
    return length




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model_class = AutoModelForCausalLM
    tokenizer_class = AutoTokenizer
    model_name_or_path = sys.argv[1]
    length = 100

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    tokenizer.padding_side = 'left'

    print(len(tokenizer), tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
    config = PeftConfig.from_pretrained(model_name_or_path)
    print(config)
    model = model_class.from_pretrained(model_name_or_path)
    #model.config.pad_token_id = tokenizer.eos_token_id
    model.to(device)
    length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)

    split_file = sys.argv[2]

    #eval_dataset으로 valid 사용
    test_path = f"/data/hzlcodus/beanpole/women/info/{split_file}.txt"
    
    # 프롬프트 생성
    prompt_text_dict = read_e2e_files(test_path, tokenizer, None) # 텍스트가 src, 여러 tgt들 이렇게 dict로 주어지는 듯
    prompt_text_lst = list(prompt_text_dict.keys()) # src들만 모아둠
    decode_mode = "sample"

    # 모델 답, gold 답들, 문제 파일 만드는 코드
    curr_dir = os.path.join('/home/hzlcodus/codes/peft/outputs', '{}_{}_{}'.format(model_name_or_path, split_file, decode_mode))
    gold_dir = os.path.join('/home/hzlcodus/codes/peft/outputs', '{}_{}_{}'.format(model_name_or_path, split_file, 'gold'))
    write_e2e_corr(prompt_text_lst, prompt_text_dict, gold_dir) # src prompt 같은 것끼리 그룹화하려고 그 사이에 한 줄씩 띄워서 gold 저장
    src_dir = os.path.join('/home/hzlcodus/codes/peft/outputs', '{}_{}_{}'.format(model_name_or_path, split_file, 'src'))
    write_e2e_src(prompt_text_lst, src_dir) # 서로 다른 src들만 쭉 저장
    out_handle = open(curr_dir, 'w')

    for prompt_idx, prompt_text in enumerate(prompt_text_lst):
        print(f"prompt_idx {prompt_idx}, prompt_text {prompt_text}")
        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt
        

        output_sequences = model.generate(
                    input_ids=input_ids,
                    emb_match=None, 
                    control_code=None,
                    max_length=length + len(encoded_prompt[0]),
                    temperature=1.0,
                    top_k=0,
                    top_p=0.9, #0.5
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                    do_sample=True, # False
                    num_beams=10,
                    no_repeat_ngram_size=4,
                    length_penalty=0.9,
                    bad_words_ids=[[628], [198]] if True else None,
                    num_return_sequences=1,
                )
    
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            # args.stop_token = tokenizer.eos_token
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            #print(text)
            text_output = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
            idx = text_output.find(tokenizer.eos_token)
            if idx >= 0:
                text_output = text_output[:idx]
            text_output = text_output.strip()

            if text_output:
                print(text_output, file=out_handle)
            else:
                print('Error', file=out_handle)
    
    print("=====FINISHED, Run Evaluate Codes=====\n", gold_dir, '\n', curr_dir)
 

if __name__ == "__main__":
    main()
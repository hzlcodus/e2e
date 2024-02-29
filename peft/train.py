from logging import Logger
import os
import pickle
import random
import time
import copy
import json
import ast
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from torch.nn.utils.rnn import pad_sequence

#from accelerate import Accelerator
from transformers import (
    TrainingArguments, Trainer, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig
)
import evaluate

import copy
import logging
from dataclasses import dataclass, field
from peft import get_peft_model, LoraConfig, TaskType
import wandb

class LineByLineData2TextTextDataset(Dataset): # jcy !!
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str, lowdata_token:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logging.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        if lowdata_token is None:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)
        else:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} {} '.format(lowdata_token, src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = []
        for ss in src_lines:
            ssl = [la.split(':')[0].strip() for la in ss.split('|')]
            # print(ssl)
            ssl_lst.append(ssl)

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []

        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1])
                self.tgt_sent.append(self.examples[i][sep_idx-1:])
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx-1
                temp_tgt_len += len(elem) - (sep_idx-1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)


        print(edited_sents[0])
        print(lines[0])

        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long), # input_ids
                torch.tensor(self.labels[i], dtype=torch.long), # input_ids에서 source는 -100으로
                torch.tensor(self.src_sent[i], dtype=torch.long), # input_ids 중 source 부분만 잘라
                torch.tensor(self.tgt_sent[i], dtype=torch.long), # input_ids 중 target 부분만 잘라 
                torch.tensor(self.src_cat[i], dtype=torch.long), # name 등 카테고리 이름들만 input_ids를 모음

                )


@dataclass
class DataCollatorForData2TextLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = True # jcy : False
    format_mode: str = 'cat'
    mlm_probability: float = 0.15

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        # print(examples[0])
        # print(len(examples))
        input_ids, labels, src, tgt, cate = zip(*examples)
        # print(len(input_ids), len(labels), len(weights))
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:

            # print(self.format_mode)
            if self.format_mode == 'cat':
                mode_input = 3
            elif self.format_mode == 'peek':
                mode_input = 1
            elif self.format_mode == 'nopeek':
                mode_input = 2
            elif self.format_mode == 'infix':
                mode_input = 4

            # mode_input = 1 # means that we take the input again.
            # mode_input = 2 # means that we do not peek at src again.
            # mode_input = 3 # means that we look at the categories, and see the input again.

            # print(self.format_mode, mode_input)

            if mode_input == 1:
                # input, batch
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
                # tgt = self._tensorize_batch(tgt)
            elif mode_input == 2:
                # nopeek.
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
            elif mode_input == 3:
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(cate)
                cate_batch, cate_attn = None, None
            elif mode_input == 4:
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)

                cate_batch = self._tensorize_batch(cate)
                cate_attn = (cate_batch != self.tokenizer.pad_token_id)

            labels[labels == self.tokenizer.pad_token_id] = -100 # tgt
            src_attn = (src != self.tokenizer.pad_token_id) # src
            tgt_attn = (batch != self.tokenizer.pad_token_id) # tgt

            if cate_batch is None:
                return {"input_ids": batch, "labels": labels,} # jcy : src에 카테고리가 들어가는 듯
            else:
                return {"input_ids": batch, "labels": labels, "cate_batch":cate_batch}

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels



'''
'meaning_representation' : name[The Vaults], eatType[pub], priceRange[more than £30], customer rating[5 out of 5], near[Café Adriatic]
'human_reference' : The Vaults pub near Café Adriatic has a 5 star rating. Prices start at £30.
'''

wandb.init(project="ssf")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

checkpoint = "EleutherAI/polyglot-ko-1.3b"
tuning_mode = 'finetune'
task_mode = 'data2text'

config = AutoConfig.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True) # jcy
config._my_arg_tune_mode = tuning_mode
config._my_arg_task_mode = task_mode
config.return_dict = True

def get_dataset(
    file_path, tokenizer
):
    dataset = LineByLineData2TextTextDataset(tokenizer=tokenizer, file_path=file_path,
                                            block_size=block_size, bos_tok=tokenizer.bos_token,
                                            eos_tok=tokenizer.eos_token,
                                            lowdata_token=None)
    return dataset


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    base_model_name_or_path="EleutherAI/polyglot-ko-1.3b"
)

model = AutoModelForCausalLM.from_pretrained( # jcy
                checkpoint,
                config=config,
            )
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

block_size = tokenizer.model_max_length = 512
print('adapting the size of the model embedding to include [PAD]')
print('len(tokenizer) = ', len(tokenizer))
num_added_tokens = tokenizer.add_special_tokens(
    {'pad_token': '[PAD]'})
embedding_layer = model.resize_token_embeddings(len(tokenizer))
print('len(tokenizer) = ', len(tokenizer))
print(tokenizer.eos_token, tokenizer.eos_token_id)
tokenizer.bos_token = tokenizer.eos_token
print(tokenizer.bos_token, tokenizer.bos_token_id)

train_file_path = "/data/hzlcodus/train_extract.txt"
#eval_file_path = "/home/hzlcodus/test.txt"

train_dataset = (
    get_dataset(file_path = train_file_path, tokenizer=tokenizer)
)

# eval_dataset = (
#     get_dataset(file_path = eval_file_path, tokenizer=tokenizer)
# )

data_collator = DataCollatorForData2TextLanguageModeling(
                tokenizer=tokenizer, mlm=False, mlm_probability=0.15,
                format_mode='cat'
            )


training_args = TrainingArguments(
    output_dir="ssf",
    evaluation_strategy="no",
    per_device_train_batch_size=1,  # Aligned with DeepSpeed
    num_train_epochs=1,
    weight_decay=0.01,  # Aligned with DeepSpeed
    gradient_accumulation_steps=8, 
    learning_rate=0.0002,  # Aligned with DeepSpeed
    warmup_steps=500,  # Aligned with DeepSpeed
    prediction_loss_only=True,
    save_steps=50000,
    deepspeed='ds_config.json'
)


args_dict = training_args.to_dict()
wandb.config.update(args_dict)

trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=None
        )

trainer.train()
model_path = "ssf-5.5-saved"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
trainer.save_model(output_dir=f"/home/hzlcodus/model/{model_path}")
# results = trainer.evaluate(eval_dataset)
# print(results)
# logging.info("Evaluation Results: %s", results)

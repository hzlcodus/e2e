import argparse
import logging

import numpy as np
import torch
import json
import sys
import openai


from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig,
    pipeline,
)
import sys, os
from peft import PeftConfig
import os
from openai import OpenAI

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# Read files from generated.txt
# read each line and ask GPT if there is a repetition

# if there is a repetition, then add the line to the list of lines to be removed
# if there is no repetition, then add the line to the list of lines to be kept

# write the lines to be kept to kept.txt
# write the lines to be removed to removed.txt

# Load a GPT-2 model
checkpoint = "gpt2"

generator = pipeline('text-generation', model=checkpoint, tokenizer=checkpoint)
client = OpenAI(
    # This is the default and can be omitted
    api_key='',
)

# Your prompt and sentence
prompt = '''
Classify whether there is a repetition in the following sentence to three groups.
1(Major repetition) 0.5(Minor repetition) 0(Clean) :

Q: The cat sits sat cat.
A: 1(Major repetition)

Q: The cat sits on the mat.
A: 0(Clean) 

Q: 
'''
sentence = "Dog goes run dog ."

# Combine the prompt and the sentence
full_prompt = f"{prompt} {sentence} A:"

# Generate a response
response = generator(full_prompt, max_length=300, num_return_sequences=1)

# Print the response
print(response)

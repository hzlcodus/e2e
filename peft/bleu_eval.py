# import nltk
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.tokenize import word_tokenize

# #nltk.download('punkt')

# # Function to read file and tokenize sentences
# def read_and_tokenize(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return [word_tokenize(line.strip()) for line in file]

# # Read and prepare the data
# reference_file = '/home/hzlcodus/PrefixTuning/data/e2e_data/src1_valid_gold.txt'
# candidate_file = '/home/hzlcodus/PrefixTuning/data/e2e_data/src1_valid_gold.txt'
# references = read_and_tokenize(reference_file)
# candidates = read_and_tokenize(candidate_file)

# # Ensure both files have the same number of lines
# if len(references) != len(candidates):
#     raise ValueError("The number of lines in both files must be the same.")

# # Calculate BLEU score
# total_score = 0
# for ref, cand in zip(references, candidates):
#     total_score += sentence_bleu([ref], cand)

# avg_bleu_score = total_score / len(candidates)
# print(f"Average BLEU Score: {avg_bleu_score}")

import nltk
import sys
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

# Function to read file content
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()
    
file_prefix = sys.argv[1]

# Read files
prompt_content = read_file(file_prefix+'_src')
generated_content = read_file(file_prefix+'_sample')
gold_content = read_file(file_prefix+'_gold')

# Split contents
prompts = prompt_content.split('\n')
generated = generated_content.split('\n')
gold_groups = gold_content.split('\n\n')
print(f'len(prompts) is {len(prompts)}')
print(f'len(prompts) is {len(generated)}')
print(f'len(prompts) is {len(gold_groups)}')

# Ensure the lengths match
assert len(prompts) == len(generated) == len(gold_groups)

# Split gold sequences into lists of sentences
gold_sequences = [group.split('\n') for group in gold_groups]

# Tokenize the sentences for BLEU calculation
gold_sequences_tokenized = [[nltk.word_tokenize(sent) for sent in group] for group in gold_sequences]
generated_tokenized = [nltk.word_tokenize(sent) for sent in generated]

# Calculate BLEU scores for each generated sentence
bleu_scores = [nltk.translate.bleu_score.sentence_bleu(reference, candidate) for reference, candidate in zip(gold_sequences_tokenized, generated_tokenized)]

# Calculate average BLEU score
average_bleu_score = sum(bleu_scores) / len(bleu_scores)

print("Average BLEU Score:", average_bleu_score)

# Calculate the maximum BLEU score for each generated sentence against its corresponding gold sentences
max_bleu_scores = []
for reference_group, candidate in zip(gold_sequences_tokenized, generated_tokenized):
    scores = [sentence_bleu([reference], candidate) for reference in reference_group]
    max_bleu_scores.append(max(scores))

# Calculate the average of these maximum BLEU scores
average_max_bleu_score = sum(max_bleu_scores) / len(max_bleu_scores)

print("Average of Maximum BLEU Scores:", average_max_bleu_score)

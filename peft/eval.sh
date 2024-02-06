#!/usr/bin/env zsh

# Define the file path variable
# MODEL_NAME="lora-filtered-1-saved"

# # FILE_PATH="/home/hzlcodus/codes/peft/outputs/${MODEL_NAME}_valid"

# # # Use the variable in your commands
# # python3 ./decode.py "${MODEL_NAME}" "valid"
# # /home/hzlcodus/codes/e2e-metrics/measure_scores.py "${FILE_PATH}_gold" "${FILE_PATH}_sample"
# # python3 ./bleu_eval.py "${FILE_PATH}" 

# FILE_PATH="/home/hzlcodus/codes/peft/outputs/${MODEL_NAME}_test"

# python3 ./decode.py "${MODEL_NAME}" "test"
# /home/hzlcodus/codes/e2e-metrics/measure_scores.py "${FILE_PATH}_gold" "${FILE_PATH}_sample"
# python3 ./bleu_eval.py "${FILE_PATH}"

python3 ./decode.py "ssf-4.3-saved" "test_ace"
python3 ./decode.py "ssf-4.3-saved" "test_extract"
#python3 ./bleu_eval.py "/home/hzlcodus/codes/peft/outputs/ssf-4.3-saved_test_extract"
# PIQA
# python -m src.eval.main \
#     --config configs/eval_config_example.json \
#     --baseline_type end_to_end \
#     --output_file end_to_end_piqa.json \
#     --dataset ./dataset/PIQA.json

# python -m src.eval.main \
#     --config configs/eval_config_example.json \
#     --baseline_type cascade \
#     --output_file cascade_piqa.json \
#     --translation_file ./mt1_translations_piqa.json \
#     --intermediate_file intermediate_results_piqa.json \
#     --dataset ./dataset/PIQA.json 

python -m src.eval.main \
    --config configs/eval_config_example.json \
    --baseline_type prompting \
    --output_file prompting_piqa.json \
    --translation_file ./mt1_translations_piqa.json \
    --intermediate_file intermediate_results_piqa.json \
    --dataset ./dataset/PIQA.json 

python -m src.eval.main \
    --config configs/eval_config_example.json \
    --baseline_type cascade \
    --output_file cascade_piqa.json \
    --translation_file ./mt1_translations_piqa.json \
    --intermediate_file intermediate_results_piqa.json \
    --dataset ./dataset/PIQA.json 

# python -m src.eval.main \
#     --config configs/eval_config_example.json \
#     --baseline_type prompting \
#     --output_file prompting_piqa_lora.json \
#     --translation_file ./mt1_translations_piqa.json \
#     --intermediate_file intermediate_results_piqa.json \
#     --dataset ./dataset/PIQA.json \
#     --lora_path /export/fs05/dzhang98/model/MT2_lora/checkpoint-4500


# MKQA

# python -m src.eval.main \
#     --config configs/eval_config_example.json \
#     --baseline_type end_to_end \
#     --output_file end_to_end_mkqa.json \
#     --dataset ./dataset/MKQA_eval.json


# python -m src.eval.main \
#     --config configs/eval_config_example.json \
#     --baseline_type cascade \
#     --output_file cascade_mkqa.json \
#     --translation_file ./mt1_translations_mkqa.json \
#     --intermediate_file intermediate_results_mkqa.json \
#     --dataset ./dataset/MKQA_eval.json 

# python -m src.eval.main \
#     --config configs/eval_config_example.json \
#     --baseline_type prompting \
#     --output_file prompting_mkqa.json \
#     --translation_file ./mt1_translations_mkqa.json \
#     --intermediate_file intermediate_results_mkqa.json \
#     --dataset ./dataset/MKQA_eval.json


# python -m src.eval.main \
#     --config configs/eval_config_example.json \
#     --baseline_type prompting \
#     --output_file prompting_mkqa_lora.json \
#     --translation_file ./mt1_translations_mkqa.json \
#     --intermediate_file intermediate_results_mkqa.json \
#     --dataset ./dataset/MKQA_eval.json \
#     --lora_path /export/fs05/dzhang98/model/MT2_lora/checkpoint-4500








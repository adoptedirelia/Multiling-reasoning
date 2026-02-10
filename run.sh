# python -m src.eval.main --config configs/eval_config_example.json --baseline_type end_to_end --output_file end_to_end.json
# python -m src.eval.main --config configs/eval_config_example.json --baseline_type cascade --output_file cascade.json
# python -m src.eval.main --config configs/eval_config_example.json --baseline_type prompting --output_file prompting.json


# python -m src.eval.main \
#     --config configs/eval_config_example.json \
#     --baseline_type end_to_end \
#     --output_file end_to_end_mkqa.json \
#     --translation_file ./dataset/MKQA.json \
#     --dataset ./dataset/MKQA.json

# python -m src.eval.main \
#     --config configs/eval_config_example.json \
#     --baseline_type cascade \
#     --output_file cascade_mkqa.json \
#     --translation_file ./dataset/MKQA.json \
#     --dataset ./dataset/MKQA.json

# python -m src.eval.main \
#     --config configs/eval_config_example.json \
#     --baseline_type prompting \
#     --output_file prompting_mkqa.json \
#     --translation_file ./dataset/MKQA.json \
#     --dataset ./dataset/MKQA.json


# python -m src.eval.output_error \
#     --config configs/eval_config_example.json \
#     --baseline_type prompting \
#     --output_file output_error_mkqa.json \
#     --dataset ./dataset/MKQA.json


python -m src.eval.input_error \
    --config configs/eval_config_example.json \
    --baseline_type prompting \
    --output_file input_error_mkqa.json \
    --dataset ./dataset/MKQA.json

# python -m src.eval.output_error \
#     --config configs/eval_config_example.json \
#     --baseline_type cascade \
#     --output_file output_error_cascade_mkqa.json \
#     --dataset ./dataset/MKQA.json


python -m src.eval.input_error \
    --config configs/eval_config_example.json \
    --baseline_type cascade \
    --output_file input_error_cascade_mkqa.json \
    --dataset ./dataset/MKQA.json
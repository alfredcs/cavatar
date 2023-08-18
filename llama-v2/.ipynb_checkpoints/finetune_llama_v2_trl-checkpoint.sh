# pip install trl
#git clone git clone https://github.com/lvwerra/trl
# git clone https://huggingface.co/datasets/timdettmers/openassistant-guanaco to /data/hf/openassistant-guanaco 

#    --train_datafile_name /data/hf/openassistant-guanaco/openassistant_best_replies_train.jsonl \
#    --test_datafile_name /data/hf/openassistant-guanaco/openassistant_best_replies_eval.jsonl \
python trl/examples/scripts/sft_trainer.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name /data/med/drugs \
    --load_in_4bit \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2


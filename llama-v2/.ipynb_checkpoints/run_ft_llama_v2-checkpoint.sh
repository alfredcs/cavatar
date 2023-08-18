#Training LLMs can be technically and computationally challenging. In this section, we look at the tools available in the Hugging Face ecosystem to efficiently train Llama 2 on simple hardware and show how to fine-tune the 7B version of Llama 2 on a single NVIDIA T4 (16GB - Google Colab). You can learn more about it in the Making LLMs even more accessible blog.
# An example command for fine-tuning Llama 2 7B on the timdettmers/openassistant-guanaco can be found below. The script can merge the LoRA weights into the model weights and save them as safetensor weights by providing the merge_and_push argument. This allows us to deploy our fine-tuned model after training using text-generation-inference and inference endpoints.
# pip install transformers>=4.31.0 --upgrade
# pip install trl 
# pip install peft
# pip install accelerate>=0.20.3
# pip install bitsandbytes>=0.39.0
python finetune_llama_v2.py \
--model_name meta-llama/Llama-2-7b-hf \
--dataset_name /data/hf/openassistant-guanaco \
--use_4bit \
--merge_and_push

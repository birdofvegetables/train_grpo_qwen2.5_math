from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)

####################################################################################################
# 1. 加载 qwen2.5 模型并设置参数

from unsloth import is_bfloat16_supported
import torch

max_seq_length = 512  # Can increase for longer reasoning traces
lora_rank = 32  # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/ai/deepseek/model_cache/unsloth/Qwen2___5-3B",
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # False, #True, # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # Reduce if out of memory
)

from unsloth.chat_templates import get_chat_template

# 设置分词器的聊天模板为 "qwen-2.5"
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

####################################################################################################
# 2. 设置 lora 参数

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=7903,
)

####################################################################################################
# 3. 加载并处理数据集

import re
from datasets import load_dataset, Dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split="train") -> Dataset:
    # data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    # 指定本地路径加载 .parquet 文件
    data = load_dataset('parquet', data_files=f'./gsm8k/main/{split}-00000-of-00001.parquet')[split]

    data = data.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })  # type: ignore
    return data  # type: ignore


dataset = get_gsm8k_questions()


# Reward functions
# 正确性奖励函数: 如果提取的回答与参考答案相同，奖励2分，否则0分
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}",
          f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


# 整数奖励函数：如果提取的回答是整数，奖励0.5分，否则0分
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# 严格格式奖励函数：回答匹配指定的严格格式，奖励0.5分，否则0分
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# 宽松格式奖励函数：回答匹配指定的宽松格式，奖励0.5分，否则0分
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# 计算 XML 标签数量
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# XML 标签数量奖励函数： <reasoning> <answer> 开头结尾共4个标签，每出现一个奖励0.125分，每多一个answer标签扣除0.001分
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


####################################################################################################
# 4. 初始化 GRPO 训练器并启动训练

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    use_vllm=True,  # use vLLM for fast inference!
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=6,  # Decrease if out of memory
    max_prompt_length=256,
    max_completion_length=200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps=500,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",  # Can use Weights & Biases
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

####################################################################################################
# 4. 做训练前后推理对比

from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)


def infer_old(question):
    text = tokenizer.apply_chat_template([{"role": "user", "content": question}, ], tokenize=False,
                                         add_generation_prompt=True)
    output = model.fast_generate([text], sampling_params=sampling_params, lora_request=None, )[0].outputs[0].text
    print('-' * 5, f"Question: {question}")
    print(output)


def infer_new(question):
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ], tokenize=False, add_generation_prompt=True)
    # 加载用 GRPO 训练的 LoRA 模型
    output = \
    model.fast_generate(text, sampling_params=sampling_params, lora_request=model.load_lora("grpo_saved_lora"), )[
        0].outputs[0].text
    print('-' * 5, f"Question: {question}")
    print(output)


question1 = "Calculate pi."
question2 = "Which is bigger? 9.919 or 9.92?"

print("----- 微调前模型推理 ------")
infer_old(question1)
infer_old(question2)

model.save_lora("grpo_saved_lora")

print("----- 微调后模型推理 ------")
infer_new(question1)
infer_new(question2)

# 合并为16bit模型
model.save_pretrained_merged("Qwen2.5-3B-GRPO-RL-gsm8k", tokenizer, save_method="merged_16bit", )

# 保存为 GGUF 模型
# model.save_pretrained_gguf("Qwen2.5-3B-GRPO-RL-gsm8k-GGUF", tokenizer,)
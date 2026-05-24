import torch
import datasets
import accelerate
import trl
device="cuda" if torch.cuda.is_available() else 'cpu'
#print({device})
model_name="openai-community/gpt2"
from transformers import AutoTokenizer, AutoModelForCausalLM
model=AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


from datasets import load_dataset
dataset=load_dataset('tatsu-lab/alpaca')


def format_sample(sample):
    instruction = sample["instruction"]
    input_text = sample["input"]
    output = sample["output"]

    if input_text.strip() != "":
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task.

### Instruction:
{instruction}

### Response:
"""

    full_text = prompt + output + tokenizer.eos_token

    return {
        "prompt": prompt,
        "full_text": full_text
    }


dataset=dataset.map(format_sample) 

dataset=dataset["train"].train_test_split(test_size=0.2,
                                          shuffle=True,
                                          seed=42)


def tokenize_function(sample):
    tokenized_full = tokenizer(
        sample["full_text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    tokenized_prompt = tokenizer(
        sample["prompt"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    input_ids = tokenized_full["input_ids"]
    labels = input_ids.copy()

    prompt_len = sum(tokenized_prompt["attention_mask"])  #sun=number of real tokens [1,1,1,0,0]

    #  IGNORE prompt tokens
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels
    }
tokenized_dataset=dataset.map(tokenize_function,batched=False)



#print(transformers.__version__)----> 5.5.3

from transformers import TrainingArguments
training_args=TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=500,
    fp16=True
)

from transformers import Trainer

trainer=Trainer(model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"]
                )

trainer.train()

trainer.save_model("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./fine_tuned_gpt2")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_gpt2")

def generate (instruction, input=""):
  if input:
    prompt=f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
  else:
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

  inputs = tokenizer(prompt, return_tensors="pt")

  outputs = model.generate(
      **inputs,
      max_new_tokens=100,
      temperature=0.7, 
      top_p=0.9,
      repetition_penalty=1.2, 
      do_sample=True 
  )

  return tokenizer.decode(outputs[0], skip_special_tokens=True)

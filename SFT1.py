import torch
import datasets
import accelerate
import trl
device="cuda" if torch.cuda.is_available() else 'cpu'

model_name="openai-community/gpt2"

from transformers import AutoTokenizer, AutoModelForCausalLM
model=AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token #Because There is no pad token in GPT-2


from datasets import load_dataset
dataset=load_dataset('tatsu-lab/alpaca')
#print(f"number of samples in the datset:{len(dataset['train'])}")----> 52002


def format_sample(sample):
  instruction=sample['instruction']
  input=sample['input']
  output=sample['output']

  if input.strip()!="":
    text=f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ###Input:
    {input}

    ###Response:
    {output}"""
    return {"text":text}
  else:
    text=f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ###Instruction:
    {instruction}

    ###Response:
    {output}"""
    return {"text":text}

dataset=dataset.map(format_sample) 

dataset=dataset["train"].train_test_split(test_size=0.2,
                                          shuffle=True,
                                          seed=42)
#dataset["train"][1]["text"]
#dataset["test"][2]["text"]

def tokenize_function(sample):
  return tokenizer(
      sample["text"],
      truncation=True,
      padding="max_length",
      max_length=512,
  )

tokenized_dataset=dataset.map(tokenize_function,batched=True)
#print(tokenized_dataset["train"][0].keys())

def add_labels(sample):
  sample["labels"]=sample["input_ids"].copy()
  return sample
tokenized_dataset=tokenized_dataset.map(add_labels)

#print(tokenized_dataset["train"][0].keys())
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
 #till here model is made and saved

#after this , model is load and generated reponse for different instruction
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


print(generate("What is Machine Learning")

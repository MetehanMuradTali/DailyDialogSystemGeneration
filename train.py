import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM

import torch

torch.cuda.empty_cache()

model_checkpoint = "t5-base"
raw_dataset = load_dataset("csv",data_files="dialogs.csv")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(row):
    model_inputs = tokenizer(row["question"], max_length=64, truncation=True)

    labels = tokenizer(text_target=row["answer"], max_length=64, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs 

raw_dataset=raw_dataset["train"].train_test_split(train_size=0.9,test_size=0.1)

print(raw_dataset)

tokenized_dataset=raw_dataset.map(preprocess_function,remove_columns=["Unnamed: 0",'question_as_int', 'answer_as_int', 'question_len', 'answer_len','question',"answer"])

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to("cuda")

batch_size = 1
args = Seq2SeqTrainingArguments(
    "output",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
torch.cuda.empty_cache()

train_results = trainer.train()
trainer.save_model("./model") 
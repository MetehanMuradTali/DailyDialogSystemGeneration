from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

def testing(text):

  new_model_path = "./model"

  tokenizer = AutoTokenizer.from_pretrained("t5-base")

  new_model = AutoModelForSeq2SeqLM.from_pretrained(new_model_path)

  inputs = tokenizer.encode(f"{text}", return_tensors="pt",max_length=64, truncation=True)

  outputs = new_model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)

  decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

  return decoded_output

#print(testing("Hi"))
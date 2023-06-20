import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

question = "What is the capital of France?"
context = "Paris is the capital and largest city of France."

inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

start_logits, end_logits = model(**inputs).values()

start_index = torch.argmax(start_logits)
end_index = torch.argmax(end_logits)

answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
answer_tokens = tokenizer.convert_ids_to_tokens(answer_tokens, skip_special_tokens=True)
answer = tokenizer.convert_tokens_to_string(answer_tokens)

print("Question:", question)
print("Answer:", answer)
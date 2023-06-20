import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

class Question(BaseModel):
    question: str
    context: str

@app.get("/")
def root():
    return {"message": "Welcome to the question answering API!"}

@app.post("/answer")

def answer_question(question: str, context: str):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

    start_logits, end_logits = model(**inputs).values()

    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)

    answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_tokens, skip_special_tokens=True)
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return {"question": question, "answer": answer}

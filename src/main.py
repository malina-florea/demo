from fastapi import FastAPI, Response
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained('readerbench/RoGPT2-base')
model = AutoModelForCausalLM.from_pretrained('readerbench/RoGPT2-base')
app = FastAPI()


class Query(BaseModel):
    ro_sentence: str


@app.get("/")
def index():
    return {'text': 'Hello World!'}


@app.get("/health")
def healthcheck():
    return Response(content='OK', media_type='text/plain', status_code=200)


@app.post("/text-generator")
def text_generator(query: Query):
    inputs = tokenizer.encode(query.ro_sentence, return_tensors='pt')
    text = model.generate(inputs, max_length=1024,  no_repeat_ngram_size=2)
    return tokenizer.decode(text[0])


# TODO: remove
# if __name__ == '__main__':
#     input_text = input('Inceput de propozitie: ')
#
#     inputs = tokenizer.encode(input_text, return_tensors='pt')
#     text = model.generate(inputs, max_length=1024,  no_repeat_ngram_size=2)
#     print(tokenizer.decode(text[0]))
#

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('readerbench/RoGPT2-base')
model = AutoModelForCausalLM.from_pretrained('readerbench/RoGPT2-base')

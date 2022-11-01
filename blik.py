from pathlib import Path
import streamlit as st
import transformers
import os
import time
import torch
import torch.nn.functional as F
from torch import nn 
from torch.cuda.amp import custom_fwd, custom_bwd
from transformers import AutoTokenizer, AutoModelForCausalLM
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import GetPosts, NewPost
from wordpress_xmlrpc.methods.users import GetUserInfo

def infer(input_ids, max_new_tokens, temperature):

    output_sequences = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )
    return output_sequences

GPT1 = "moyix/codegen-6B-multi-gptj"
GPT2 = "moyix/codegen-16B-multi-gptj"
GPT3 = "moyix/codegen-2B-multi-gptj"

#Prompts
st.title("Generate code with GPT codegen 🦄")
st.subheader("This machine learning model generates code in every language for your development projects")

generate = st.button("GENERATE CODE")
text_target = st.text_area(label = "Enter your instruction and let the answer open for the generated code, you can choose between a 6, 16 and 2 parameter model (it can take some time to generate)", value ="""Instruction: Generate a GAN that can generate new NFT's from a dataset
Answer:""", height = 300)
models = st.selectbox('Choose model that you want to run', [GPT1, GPT2, GPT3])
max_new_tokens = st.slider("Max Length", value = 500, min_value = 100, max_value=1000, step=50)
temperature = st.slider("Temperature", value = 0.9, min_value = 0.0, max_value=1.0, step=0.1)
batch_size = 1

if torch.cuda.is_available():
	dev = 'cuda'
else:
	dev = 'cpu'

def load_model():
    device = torch.device(dev)
    model = AutoModelForCausalLM.from_pretrained(models)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B") and AutoTokenizer.from_pretrained("Salesforce/codegen-2B-multi") and AutoTokenizer.from_pretrained("Salesforce/codegen-6B-multi") and AutoTokenizer.from_pretrained("Salesforce/codegen-16B-multi")
    return model, tokenizer, device

#Load model and generate
if generate:
    torch.cuda.empty_cache()
    with st.spinner("AI is at work......"):
        latest_iteration = st.empty()
        bar = st.progress(0)
        bar.progress(5)
        model, tokenizer, device = load_model()
        bar.progress(20)
        input_ids = tokenizer(text=text_target, return_tensors="pt").input_ids
        bar.progress(50)
        output_sequences = infer(input_ids, max_new_tokens, temperature)
        bar.progress(70)
        generated_ids = model.generate(output_sequences)
        bar.progress(90)
        generated_text = tokenizer.decode(generated_ids[0])
        wp = Client('https://yourwordpresswebsite.com/xmlrpc.php', 'Username', 'Password')
        post = WordPressPost()
        post.title = text_target
        post.content = """[code language="actionscript3"]""" + generated_text + """[/code]"""
        post.post_status = 'publish'
        wp.call(NewPost(post))
        bar.progress(100)
    st.success("AI Succesfully generated code")
    print(generated_text) 

    st.code(generated_text)

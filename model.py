import streamlit as st
import os
import transformers

import torch
import torch.nn.functional as F
from torch import nn 
from torch.cuda.amp import custom_fwd, custom_bwd
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("moyix/codegen-16B-multi-gptj", low_cpu_mem_usage=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B") and AutoTokenizer.from_pretrained("Salesforce/codegen-16B-multi")
    return model, tokenizer, device
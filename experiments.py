# set home directory
home_directory = ""
llama_token = ""


import sklearn
import transformers
import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import csv
import termcolor
import torch
import scipy.optimize
from termcolor import colored
from scipy.spatial import distance
from sklearn.manifold import MDS
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from transformers import pipeline, AutoTokenizer, AutoModel
from transformers import GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXModel, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from multiprocessing import Pool
import time
import os

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from effects.numeric_capabilities import numeric_effects_main
#from effects.typicality import typicality_main
#from effects.similarity_between_number_and_non_number_words import sim_num_nonNum_main
#from effects.ravens import ravens_main

#from effects.evaluate_raven import evaluate_ravens_main
#from effects.sat_turney_items import sat_tests_main
#from effects.run_lm_eval import lm_eval_main
compare_list = ["meta-llama/Llama-2-7b-hf", 
"meta-llama/Llama-2-13b-hf", 
#"berkeley-nest/Starling-LM-7B-alpha",
 #"tiiuae/falcon-7b", 
 "mistralai/Mistral-7B-v0.1", 
 #"mistralai/Mistral-7B-Instruct-v0.2", 

 "Qwen/Qwen1.5-0.5B", 
 "Qwen/Qwen1.5-1.8B", 
 "Qwen/Qwen1.5-4B", 
 "Qwen/Qwen1.5-7B", 
# "Qwen/Qwen1.5-14B"]
#sizes = [32, 40, 32, 32, 32, 32, 24, 24, 40, 32, 40]
sizes = [32, 40, 32, 24, 24, 40, 32]


#model_names = ["EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-160m-deduped", "EleutherAI/pythia-410m-deduped", "EleutherAI/pythia-1b-deduped", "EleutherAI/pythia-1.4b-deduped", "EleutherAI/pythia-2.8b-deduped", "EleutherAI/pythia-6.9b-deduped", "EleutherAI/pythia-12b-deduped"]
model_names = ["EleutherAI/pythia-70m-deduped"]

model_hidden_states = [6]
model_names = model_names + compare_list
model_hidden_states = model_hidden_states + sizes

def load_model(model_name, revision, device):

    if model_name == "LLM360/Amber":
        tokenizer = LlamaTokenizer.from_pretrained("LLM360/Amber", 
        output_hidden_states  = True,
        revision=revision)
        model = LlamaForCausalLM.from_pretrained("LLM360/Amber",
        output_hidden_states  = True,
         revision=revision)


    elif model_name in compare_list:
         
        if model_name ==  "meta-llama/Llama-2-70b-hf":
            tokenizer = AutoTokenizer.from_pretrained(model_name, 
            torch_dtype=torch.float16, token="")
            model = AutoModelForCausalLM.from_pretrained(model_name,
            output_hidden_states  = True,
             torch_dtype=torch.float16, load_in_8bit=True, token=llama_token)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, 
            torch_dtype=torch.float16, token=llama_token)
            model = AutoModelForCausalLM.from_pretrained(model_name,
            output_hidden_states  = True,
            torch_dtype=torch.float16, token=llama_token)
    else:

        model = GPTNeoXForCausalLM.from_pretrained( 
            model_name, 
            revision=revision,
            output_hidden_states  = True,
            
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision
            ) 
    model.to(device)
    return model, tokenizer

def multiproc(model_name, revision, model_hidden_state, directory, device):
    start = time.time()
    if model_name!="LLM360/Amber":
        revision = "step"+ revision
    model = None
    tokenizer = None
    model, tokenizer = load_model(model_name, revision, device)
    end = time.time()

    print("----TIME (s): /experiments/model-loaded---", end - start)
    #ravens_main(model, tokenizer, model_hidden_state, directory, revision, device, model_name)
    #evaluate_ravens_main(model, tokenizer, model_hidden_state, directory, revision, device, model_name)
    #sat_tests_main(model, tokenizer, model_hidden_state, directory, revision, device, model_name)
    #lm_eval_main(model, tokenizer, model_hidden_state, directory, revision, device, model_name)
    #typicality_main(model, tokenizer, model_hidden_state, directory, revision, device, model_name)
    numeric_effects_main(model, tokenizer, model_hidden_state, directory, revision, device, model_name)
    end = time.time()
    print("----TIME (s): /experiments/all-experiments-done---", end - start)
    return 

def get_pythia_revision_steps():
    list_a = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] 
    list_b = [i for i in range(1000, 143000, 1000)] 
    list_c = list_a + list_b
    steps = [str(i) for i in list_c]
    return steps

def model_experiments(model_name, model_hidden_state, device):
    # make folder here for model_name
    directory = "outputs/" + model_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    
   

    steps = get_pythia_revision_steps()


    if model_name=="LLM360/Amber":
        steps = ["ckpt_"+f"{i:03}" for i in range(0,20)] 
    elif model_name in compare_list:
        steps = ["1"]
    inputs = [(model_name, revision, model_hidden_state, directory, device) for revision in steps]
    for i in inputs:
        print(i)
        multiproc(i[0], i[1], i[2], i[3], i[4])





if __name__ == "__main__":
    device = torch.device("cuda")
    for i in range(len(model_names)):
        model_experiments(model_names[i], model_hidden_states[i], device)

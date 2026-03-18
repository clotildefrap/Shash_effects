# required imports
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
from transformers import GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXModel
import os
import math

def make_vector_pythia(text, model, tokenizer, device):
  a1 = text
  b1 = tokenizer.encode(a1)
  input_ids = torch.tensor(b1).unsqueeze(0).to(device)  # Batch size 1
  outputs = model(input_ids)
  vector_array = []
  hidden_states = [i for i in range(1,len(outputs.hidden_states))]

  for hidden_state in hidden_states:

    last_hidden_states = outputs.hidden_states[hidden_state]
    if len(last_hidden_states.shape)>1: # usually will have a batch, seq_lenght and hidden_states
      last_hidden_states = torch.mean(last_hidden_states, 1)
      #The 1 specifies the dimension (axis) over which you take the mean : probably the seq_length
    last_hidden_states = last_hidden_states[0].flatten() #if the first is batch, then keep only hidden_states
    vectors = last_hidden_states.cpu().detach().numpy() # move tensor from GPU → CPU, remove gradient tracking, convert to NumPy array
    vector_array.append(vectors)
  return vector_array



def best_fit_line(X, Y):
  X = list(X)
  xbar = sum(X)/len(X)
  ybar = sum(Y)/len(Y)
  n = len(X) # or len(Y)
  numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar # this is a form of covariance, checks of xi and yi move together and centers the data
  denum = sum([xi**2 for xi in X]) - n * xbar**2 # related to the variance of X
  b = numer / denum # slope of the best fitting line
  a = ybar - b * xbar # intercept
  return a, b


def normalize(arr, t_min = 0, t_max = 1, test = False):
    if test ==False:
      return arr
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    if(diff_arr != 0):
      for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def distance_effect(list_numbers, vectors, hidden_state = -1):
  dict_1 = {}
  list_1 = []
  for i in range(len(list_numbers)):
    for j in range(i , len(list_numbers)):
      if((j-i) not in dict_1):

        dict_1[j-i] = [[i+1,j+1,1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )]]
        # takes in vectors (dict or list) a specific item and select a specific vector in it
        # it computes the cosine distance by doing 1 - cosine similarity. 
        # we get the similarity by doing 1 - cosine distance [-1 , 1]
        #first two elements are human friendly indices. The key is the distance ebtween the numbers so
        # all the pairs with the same distance are stored together. 
        list_1.append([i+1,j+1,1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])
      else:
        dict_1[j-i].append([i+1,j+1,1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])
        list_1.append([i+1,j+1,1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])
  dict_2 = {}
  for key in dict_1:
    lists = dict_1[key]
    sum = 0
    for x in lists:
      sum+=x[2] # 2 = cosine similarity, sums all cosine similarities for this distance
    avg = sum/len(lists)
    dict_2[key] = avg
    y1 = list(sorted(lists, key=lambda lists: lists[2], reverse = True)) # dict_2 = {}
  distance_effect = []
  for i in range(len(dict_2)):
    distance_effect.append(dict_2[i]) # converts into a list
  return distance_effect[1:]


def mds_funct(list_numbers, vectors, hidden_state = -1):
  dict_1 = {}
  list_1 = []
  l_2d_ok = [[0] * 8 for i in range(8)] # creates an 8×8 2D list (list of lists) filled with zeros
  for i in range(len(list_numbers)):
    for j in range(i, len(list_numbers)):
      if((j-i) not in dict_1):
        dict_1[j-i] = [[i+1,j+1,1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )]]
        list_1.append([i+1,j+1,1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])
      else:
        dict_1[j-i].append([i+1,j+1,1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])
        list_1.append([i+1,j+1,1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])
  for i in list_1:
    l_2d_ok[i[0]-1][i[1]-1] = 1-i[2] # i[0]-1 and i[1]-1 → convert from 1-based indexing to Python’s 0-based indexing.
    l_2d_ok[i[1]-1][i[0]-1] = 1-i[2] # 1 - i[2] → turns similarity back into a "distance-like" value.
  # we make the matrix symmetrical, so that the lower triangle is also filled
  # print(l_2d_ok)
  return l_2d_ok

def normalize_size(arr, size_effect_avg, t_min = 0, t_max = 1):
    norm_arr = []
    diff = t_max - t_min
    max1 = 0
    min1 = 10000
    for i in arr:
      max1 = max(max1, max(i))
      min1 = min(min1, min(i))
    diff_arr = max1- min1
    for j in arr:
      art = []
      for i in j:
        temp = (((i - min1)*diff)/diff_arr) + t_min # i - min1 → shifts the value so the min becomes 0.
        #(i - min1) * diff / diff_arr → scales it to the new range size diff.
        # + t_min → shifts the minimum to the target minimum t_min.
        art.append(temp)    # add the normalized value to the current row art.
      norm_arr.append(art)
    norm_arr_avg = []
    for i in size_effect_avg:
        temp = (((i - min1)*diff)/diff_arr) + t_min
        norm_arr_avg.append(temp)
    return norm_arr,norm_arr_avg


def to_full(a):
    output = np.full((len(a), max(map(len, a))), np.nan) # list of lists with different lengths into a rectangular NumPy array, adds Nan in empty spaces
    # len(a) = number of rows, map(len, a) = gets the length of each sublist, max(...) = finds the longest row
    for i, row in enumerate(a):
        output[i, :len(row)] = row # select row i, :len(row) → only fill up to the length of that row
    return output

def size_effect(list_numbers, vectors, hidden_state = -1):
  dict_1 = {}
  list_1 = []
  for i in range(len(list_numbers)):
    for j in range(i, len(list_numbers)):
      if((j-i) not in dict_1):
        dict_1[j-i] = [[i+1,j+1,1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )]]
        list_1.append([i+1,j+1,1 -  distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])
      else:
        dict_1[j-i].append([i+1,j+1, 1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])
        list_1.append([i+1,j+1, 1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])

  dict_2 = {}
  size_effect = []
  for i in dict_1:
    size_effect.append([ k[2] for k in dict_1[i]])
  size_effect_averaged = to_full(np.array(size_effect[1:],dtype=object)).T # = transpose
  # This allows lists of different lengths (ragged lists). Without dtype=object, NumPy would complain.
  # to_full : It converts ragged lists into a rectangular array with NaN padding.
  # axis=1 → compute mean across each row. nanmean → ignores NaN values
  size_effect_averaged = np.nanmean(size_effect_averaged, axis=1)
  return size_effect[1:], size_effect_averaged

def ratios(list_numbers, vectors, hidden_state = -1):
  dict_1 = {}
  list_1 = []
  for i in range(len(list_numbers)):
    for j in range(i, len(list_numbers)):
      if((j-i) not in dict_1):
        dict_1[j-i] = [[i+1,j+1,1 -distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )]]
        list_1.append([i+1,j+1,1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])
      else:
        dict_1[j-i].append([i+1,j+1,1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])
        list_1.append([i+1,j+1, 1 - distance.cosine(vectors[list_numbers[i]][hidden_state], vectors[list_numbers[j]][hidden_state] )])

  t1 = sorted(list_1, key=lambda list_1: list_1[2], reverse = True)
  y = [a[2] for a in t1 if a[2]!=1 ] # Take only the 3rd value (a[2]), Ignore values equal to 1 (probably perfect similarity)
  y = normalize(y)
  if(np.isnan(y[0])):
    print(list_numbers, hidden_state)
  x = range(len(y))
  tex = [str((a[0],a[1])) for a in t1 if a[2]!=1 ] # creates labels for each pairs
  tex2 = [a[1]/a[0] for a in t1 if a[2]!=1 ] # get the ration of the one over the other
 
  xs = np.array(tex2)
  ys = np.array(y)

  params, cv = scipy.optimize.curve_fit(lambda t, a, b, c: a * np.exp(-b * t) + c, xs, ys,maxfev=1000000)
  # tries to find the best parameters a,b,c for the fucntion to fit the data points
  a, b, c = params
  #a → starting height, b → how fast it decays, c → the minimum value it approaches
  x_fitted = np.linspace(np.min(xs), np.max(xs), 100)
  #Generates 100 evenly spaced points between min and max of xs --> To draw a smooth curve instead of just points

  y_fitted = a * np.exp(-b * x_fitted) + c # y_fitted = a * np.exp(-b * x_fitted) + c
  squaredDiffs = np.square(ys - (a * np.exp(-b * xs) + c)) # squares the difference between actual values and predicted ones = error²
  squaredDiffsFromMean = np.square(ys - np.mean(ys)) # Measures how far data is from its average = total variability in your data
  rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
  # R² = 1 → perfect fit, R² = 0 → model is as bad as just using the mean, R² < 0 → model is worse than the mean 
  return tex2, y, tex, rSquared, x_fitted, y_fitted, params
  # tex2 → ratios (j/i), y → normalized values, tex → labels like (i, j)
  # rSquared → quality of fit, x_fitted → smooth x values, y_fitted → smooth curve values, params → (a, b, c)
  

def get_vectors_for_all_numbers(texts, model, tokenizer, device):
  dict_of_vectors = {}
  for i in texts: 
    dict_of_vectors[i] = make_vector_pythia(i, model,  tokenizer, device)

  return dict_of_vectors


def run_all(all_text, model, tokenizer, device):
    
  vectors = get_vectors_for_all_numbers(all_text, model, tokenizer, device)
  return vectors



def distance_effect_main(model, tokenizer, vectors, list_numbers_1, list_numbers_2, list_numbers_3, model_hidden_state):
  list_of_rs = []
  list_range = []
  list_max = []
  list_rsqr = []
  for hidden_state in range(0, model_hidden_state): 
    effects = []
    rs = []
    ranges = []
    maxes = []
    effects.append(distance_effect(list_numbers_1, vectors, hidden_state = hidden_state))
    effects.append(distance_effect(list_numbers_2, vectors, hidden_state = hidden_state))
    effects.append(distance_effect(list_numbers_3, vectors, hidden_state = hidden_state))
    model_type = ['input: Lowercase number words', 'input: Uppercase number words', 'input: Numerical Values']
    model_type_1 = ['input: Lowercase number words', 'input: Uppercase number words', 'input: Numerical Values', 'Average']
    for eff in range(len(effects)):
      distance_effect_outs = effects[eff]
      X = range(1, len(distance_effect_outs)+1)
      Y = normalize(distance_effect_outs)
      Y = distance_effect_outs
      a, b = best_fit_line(X, Y)
      yfit = [a + b * xi for xi in X] # gives predicted values from the best_fit_line
      variance = np.var(Y) # Measures how spread out your data is.
      residuals = np.var([(b*xx + a - yy)  for xx,yy in zip(X,Y)])
      Rsqr = 1-residuals/variance # R²
      rs.append(Rsqr)
      ranges.append(max(Y)-min(Y)) # spread of values
      maxes.append(max(Y))  
    list_of_rs.append(rs)
    list_range.append(ranges)
    list_max.append(maxes)
  df_distance_rsqr = pd.DataFrame(list_of_rs, columns =model_type, dtype = float)
  df_distance_range = pd.DataFrame(list_range, columns =model_type, dtype = float)
  df_distance_max = pd.DataFrame(list_max, columns =model_type, dtype = float)
  return df_distance_rsqr, df_distance_range, df_distance_max


def size_effect_main(model, tokenizer, vectors, list_numbers_1, list_numbers_2, list_numbers_3, model_hidden_state):
  list_of_rs = []

  for hidden_state in range(0, model_hidden_state): 
    effects = []
    rs = []
    ranges = []
    maxes = []
    effects.append(size_effect(list_numbers_1, vectors, hidden_state = hidden_state))
    effects.append(size_effect(list_numbers_2, vectors, hidden_state = hidden_state))
    effects.append(size_effect(list_numbers_3, vectors, hidden_state = hidden_state))
    model_type = ['input: Lowercase number words', 'input: Uppercase number words', 'input: Numerical Values']
    model_type_1 = ['input: Lowercase number words', 'input: Uppercase number words', 'input: Numerical Values', 'Average']
    for eff in range(len(effects)):
      size_effect_outs = effects[eff][0]
      size_effect_avg = effects[eff][1]
      X = range(1, len(size_effect_avg)+1)
      Y, size_effect_avg = normalize_size(size_effect_outs, size_effect_avg)
      a, b = best_fit_line(X, size_effect_avg)
      yfit = [a + b * xi for xi in X]
      variance = np.var(size_effect_avg)
      residuals = np.var([(b*xx + a - yy)  for xx,yy in zip(X,size_effect_avg)])
      Rsqr = 1-residuals/variance
      rs.append(Rsqr)
    list_of_rs.append(rs)
  df_size_rsqr = pd.DataFrame(list_of_rs, columns = model_type, dtype = float)
  return df_size_rsqr


def ratio_effect_main(model, tokenizer, vectors, list_numbers_1, list_numbers_2, list_numbers_3, model_hidden_state):
  list_of_rs = []

  for hidden_state in range(0, model_hidden_state): 
    effects = []
    rs = []
    ranges = []
    maxes = []
    effects.append(ratios(list_numbers_1, vectors, hidden_state = hidden_state))
    effects.append(ratios(list_numbers_2, vectors, hidden_state = hidden_state))
    effects.append(ratios(list_numbers_3, vectors, hidden_state = hidden_state))
    model_type = ['input: Lowercase number words', 'input: Uppercase number words', 'input: Numerical Values']
    model_type_1 = ['input: Lowercase number words', 'input: Uppercase number words', 'input: Numerical Values', 'Average']
    for eff in range(len(effects)):
      ratios_outs = effects[eff]
      Y = ratios_outs[1]
      X = range(len(Y))
      rs.append(ratios_outs[3])
    list_of_rs.append(rs)
  df_ratio_rsqr = pd.DataFrame(list_of_rs, columns = model_type, dtype = float)
  return df_ratio_rsqr


def mds_effect_main(model, tokenizer, vectors, list_numbers_1, list_numbers_2, list_numbers_3, model_hidden_state):
  list_of_stresses = []
  list_of_coors = []
  list_numbers_4 = [1, 2, 3, 4, 5, 6, 7, 8]
  list_numbers_5 = [math.log10(x) for x in list_numbers_4]
  for hidden_state in range(0, model_hidden_state): 
    effects = []
    stresses = []
    coors = []
    effects.append(mds_funct(list_numbers_1, vectors, hidden_state = hidden_state))
    effects.append(mds_funct(list_numbers_2, vectors, hidden_state = hidden_state))
    effects.append(mds_funct(list_numbers_3, vectors, hidden_state = hidden_state))
    model_type = ['input: Lowercase number words', 'input: Uppercase number words', 'input: Numerical Values']
    model_type_1 = ['input: Lowercase number words', 'input: Uppercase number words', 'input: Numerical Values', 'Average']
    for X in range(len(effects)):
      mds = MDS(random_state = 0, n_components = 1, metric = False,  dissimilarity='precomputed',  normalized_stress = True)
      X_transform_1 = mds.fit_transform(effects[X])
      X_transform_1 = [i[0] for i in X_transform_1]
      if(X_transform_1[0]>0):
        X_transform_1 = [-1*i for i in X_transform_1]
      X_transform = normalize(X_transform_1, -5, 5, test= True)
      X_transform_x = normalize(X_transform_1, 0, 1, test= True)
      stress = mds.stress_
      stresses.append(stress)
      correlation = np.corrcoef(X_transform_x,list_numbers_5)[0][1]
      coors.append(correlation)
    sum_1 = sum(stresses)/len(stresses)
    stresses.append(sum_1)
    list_of_stresses.append(stresses)

    sum_1 = sum(coors)/len(coors)
    coors.append(sum_1)
    list_of_coors.append(coors)
  df_stress = pd.DataFrame(list_of_stresses, columns =model_type_1, dtype = float)
 

  df_coors = pd.DataFrame(list_of_coors, columns =model_type_1, dtype = float)
 
  return df_stress, df_coors

def numeric_effects_main(model, tokenizer, model_hidden_state, directory, revision, device, model_name):
  

  list_numbers_1 = ["one", "two", "three", "four", "five", "six", "seven", "eight" ]
  list_numbers_2 = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"]
  list_numbers_3 = ["1", "2", "3", "4", "5", "6", "7", "8"]
  all_text = list_numbers_1 + list_numbers_2 + list_numbers_3

  vectors = run_all(all_text, model, tokenizer, device)
  model_hidden_state = 1
      
  df_distance_rsqr, df_distance_range, df_distance_max = distance_effect_main( model, tokenizer, vectors, list_numbers_1, list_numbers_2, list_numbers_3, model_hidden_state)
  df_size_rsqr = size_effect_main( model, tokenizer, vectors, list_numbers_1, list_numbers_2, list_numbers_3, model_hidden_state)
  df_ratio_rsqr = ratio_effect_main( model, tokenizer, vectors, list_numbers_1, list_numbers_2, list_numbers_3, model_hidden_state)
  df_stress, df_coors =  mds_effect_main( model, tokenizer, vectors, list_numbers_1, list_numbers_2, list_numbers_3, model_hidden_state)
  directory = directory +"/" + revision + "/"
  if not os.path.exists(directory):
        os.makedirs(directory)
   
  df_coors.to_excel(directory + "df_coors.xlsx")
  df_stress.to_excel(directory + "df_stress.xlsx")
  df_distance_rsqr.to_excel(directory + "df_distance_rsqr.xlsx")
  df_size_rsqr.to_excel(directory + "df_size_rsqr.xlsx")
  df_ratio_rsqr.to_excel(directory + "df_ratio_rsqr.xlsx")
  df_distance_range.to_excel(directory + "df_distance_range.xlsx")
  df_distance_max.to_excel(directory + "df_distance_max.xlsx")


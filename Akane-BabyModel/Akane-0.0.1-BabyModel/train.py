import os 
from dataset_loader import load_text_from_folder
from tokenizer import WordTokenizer
from model import BabyTransformer 
import torch

script_dir=os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join (script_dir,"Akane_dataset")

print ("Looking for dataset at:",dataset_path)
if not os.path.exists(dataset_path):
    raise FileNotFoundError (f"Dataset folder not found : { dataset_path}")

lines =load_text_from_folder(dataset_path)
print ("Loading lines ",len (lines))

tokenizer = WordTokenizer()
tokenizer.build_vocab(lines)

vocab_size = len(tokenizer.vocab)

print ("Vocab size",vocab_size)
print ("Model ready !")               

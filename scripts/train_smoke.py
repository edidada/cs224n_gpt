"""Real GPT-2 + course data integration; bounded smoke, not benchmark training."""
import csv
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from models.gpt2 import GPT2Model
from optimizer import AdamW
from datasets import load_paraphrase_data, paraphrase_prompt, SonnetsDataset

torch.manual_seed(11711)
random.seed(11711)
torch.set_num_threads(2)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
results = {}

def encode(texts):
    return tokenizer(texts, return_tensors='pt',padding=True,truncation=True,max_length=64)

def classification(name,train,dev,classes):
    model = GPT2Model.from_pretrained()
    head = torch.nn.Linear(model.config.hidden_size,classes)
    # Linear probes exercise real representations without expensive full-model CPU training.
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    opt = AdamW(head.parameters(),lr=1e-3)
    def features(rows):
        with torch.no_grad():
            x = encode([r[0] for r in rows])
            return model(x['input_ids'],x['attention_mask'])['last_token']
    train,dev = train[:16],dev[:16]
    x,y = features(train),torch.tensor([r[1] for r in train])
    xd,yd = features(dev),torch.tensor([r[1] for r in dev])
    initial = F.cross_entropy(head(x),y).item()
    for _ in range(20):
        opt.zero_grad(); loss=F.cross_entropy(head(x),y); loss.backward(); opt.step()
    final = F.cross_entropy(head(x),y).item()
    assert final < initial
    results[name] = dict(train_examples=len(train),dev_examples=len(dev),initial_train_loss=initial,final_train_loss=final,dev_accuracy=(head(xd).argmax(-1)==yd).float().mean().item(),mode='frozen-backbone smoke')

for name in ['sst','cfimdb']:
    def read(split):
        with open(f'data/ids-{name}-{split}.csv',encoding='utf-8') as f:
            return [(r['sentence'],int(r['sentiment'])) for r in csv.DictReader(f,delimiter='\t')]
    classification(name,read('train'),read('dev'),5 if name=='sst' else 2)

def pairs(split):
    return [(paraphrase_prompt(a,b),label) for a,b,label,_ in load_paraphrase_data(f'data/quora-{split}.csv')]
classification('paraphrase',pairs('train'),pairs('dev'),2)
model = GPT2Model.from_pretrained().train()
opt = AdamW(model.parameters(),lr=1e-5)
sonnets = SonnetsDataset('data/sonnets.txt')
x = encode([sonnets[0][1]])
labels=x['input_ids'][:,1:].masked_fill(x['attention_mask'][:,1:]==0,-100)
losses=[]
for _ in range(2):
    opt.zero_grad()
    logits=model.hidden_state_to_token(model(x['input_ids'],x['attention_mask'])['last_hidden_state'])[:,:-1]
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
    losses.append(loss.item())
results['sonnet'] = dict(train_examples=1,steps=2,losses=losses,mode='full-backbone smoke')
from sonnet_generation import SonnetGPT
sampler = SonnetGPT.__new__(SonnetGPT)
torch.nn.Module.__init__(sampler)
sampler.gpt = model.eval()
sampler.tokenizer = tokenizer
prompt = SonnetsDataset('data/sonnets_held_out_dev.txt')[0][1]
encoding = tokenizer(prompt,return_tensors='pt')['input_ids']
Path('results').mkdir(exist_ok=True)
for k in [0, 20]:
    torch.manual_seed(11711)
    _, text = sampler.generate(encoding,temperature=0.8,top_p=0.9,top_k=k,max_length=32)
    Path(f'results/sonnet-top-k-{k}.txt').write_text(text,encoding='utf-8')
Path('results/smoke.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))


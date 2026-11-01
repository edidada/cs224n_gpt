"""Measure real CPU full-model steps; estimates are not experiment results."""
import argparse, csv, json, os, platform, resource, statistics, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from models.gpt2 import GPT2Model
from optimizer import AdamW
from datasets import SonnetsDataset, load_paraphrase_data, paraphrase_prompt
parser=argparse.ArgumentParser(); parser.add_argument('--task',choices=['sst','cfimdb','paraphrase','sonnet'],required=True)
a=parser.parse_args(); torch.set_num_threads(2)
tokenizer=GPT2Tokenizer.from_pretrained('gpt2'); tokenizer.pad_token=tokenizer.eos_token
if a.task in ['sst','cfimdb']:
    with open(f'data/ids-{a.task}-train.csv',encoding='utf-8') as f: rows=list(csv.DictReader(f,delimiter='\t'))
    texts=[r['sentence'] for r in rows]; classes=5 if a.task=='sst' else 2
elif a.task=='paraphrase':
    rows=load_paraphrase_data('data/quora-train.csv'); texts=[paraphrase_prompt(x[0],x[1]) for x in rows]; classes=2
else:
    rows=SonnetsDataset('data/sonnets.txt'); texts=[x[1] for x in rows]; classes=None
# Probe low/median/high lengths from 256 deterministically spaced examples.
indices=list(range(0,len(texts),max(1,len(texts)//256)))[:256]
lengths=tokenizer([texts[i] for i in indices],truncation=True,max_length=1024)['input_ids']
ordered=sorted(zip(indices,lengths),key=lambda x:len(x[1]))
chosen=[ordered[int((len(ordered)-1)*q)] for q in [0.1,0.3,0.5,0.7,0.9]]
model=GPT2Model.from_pretrained().train(); head=torch.nn.Linear(model.config.hidden_size,classes) if classes else None
opt=AdamW(list(model.parameters())+(list(head.parameters()) if head else []),lr=1e-5)
timings=[]
for i,ids in chosen:
    x=torch.tensor([ids]); mask=torch.ones_like(x)
    start=time.perf_counter(); opt.zero_grad(set_to_none=True); out=model(x,mask)
    if head: loss=F.cross_entropy(head(out['last_token']),torch.tensor([0]))
    else:
        logits=model.hidden_state_to_token(out['last_hidden_state'])[:,:-1]
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1))
    loss.backward(); opt.step()
    timings.append(dict(tokens=len(ids),seconds=time.perf_counter()-start))
mean=statistics.mean(x['seconds'] for x in timings)
result=dict(task=a.task,train_examples=len(texts),cpu=platform.processor(),logical_cpus=os.cpu_count(),torch_threads=torch.get_num_threads(),cuda_available=torch.cuda.is_available(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,steps=timings,estimated_training_hours_for_10_epochs=mean*len(texts)*10/3600,estimate_excludes_evaluation_and_checkpoint_io=True)
Path('results').mkdir(exist_ok=True); Path(f'results/capacity-{a.task}.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

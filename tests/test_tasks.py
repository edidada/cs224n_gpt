from types import SimpleNamespace
from unittest.mock import patch
import torch
import pytest
from transformers import GPT2Config, GPT2Model as Reference
from models.gpt2 import GPT2Model
from classifier import GPT2SentimentClassifier
from paraphrase_detection import ParaphraseGPT
from sonnet_generation import SonnetGPT
from datasets import ParaphraseDetectionDataset, ParaphraseDetectionTestDataset

def tiny():
    return GPT2Model.from_hf_model(Reference(GPT2Config(vocab_size=41,n_positions=24,n_embd=24,n_layer=1,n_head=4)))

def test_task_heads_train():
    args = SimpleNamespace(num_labels=5, fine_tune_mode='full-model',hidden_dropout_prob=0,model_size='gpt2',d=24,l=1,num_heads=4)
    ids = torch.tensor([[1,2,3],[4,5,6]])
    for cls, classes in [(GPT2SentimentClassifier,5),(ParaphraseGPT,2)]:
        with patch.object(GPT2Model,'from_pretrained',side_effect=lambda **kw: tiny()):
            model = cls(args)
        before = model(ids,torch.ones_like(ids))
        assert before.shape == (2,classes)
        loss = torch.nn.functional.cross_entropy(before,torch.tensor([0,1]))
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gpt.parameters())

def test_generation_preserves_prompt_and_limits():
    model = SonnetGPT.__new__(SonnetGPT); torch.nn.Module.__init__(model)
    model.gpt = tiny().eval()
    model.tokenizer = SimpleNamespace(eos_token_id=40,decode=lambda x: str(x))
    ids = torch.tensor([[1,2,3]])
    result,text = model.generate(ids,max_length=0)
    assert torch.equal(result,ids) and text == '[1, 2, 3]'
    result,_ = model.generate(ids,max_length=40,top_k=1)
    assert result.shape[1] <= 24
    with pytest.raises(ValueError): model.generate(ids,temperature=0)

def test_paraphrase_labels_and_prompts():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            self.texts = texts
            return dict(input_ids=[[1,2]]*len(texts),attention_mask=[[1,1]]*len(texts))
    train = ParaphraseDetectionDataset.__new__(ParaphraseDetectionDataset); train.tokenizer = Tokenizer()
    test = ParaphraseDetectionTestDataset.__new__(ParaphraseDetectionTestDataset); test.tokenizer = Tokenizer()
    batch = train.collate_fn([('a','b',1,'id')]); test.collate_fn([('a','b','id')])
    assert batch['labels'].tolist() == [1]
    assert train.tokenizer.texts == test.tokenizer.texts

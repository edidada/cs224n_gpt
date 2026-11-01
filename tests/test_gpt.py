import torch
from transformers import GPT2Config, GPT2Model as Reference
from models.gpt2 import GPT2Model

def test_reference_and_causality():
    torch.manual_seed(42)
    ref = Reference(GPT2Config(vocab_size=41, n_positions=24, n_embd=24, n_layer=2, n_head=4, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()
    model = GPT2Model.from_hf_model(ref).eval()
    ids = torch.tensor([[1,2,3,4,5],[6,7,8,0,0]])
    mask = torch.tensor([[1,1,1,1,1],[1,1,1,0,0]])
    actual = model(ids, mask)
    expected = ref(ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(actual['last_hidden_state'], expected, atol=2e-6, rtol=2e-5)
    changed = ids.clone(); changed[:, 3:] = 9
    torch.testing.assert_close(model(changed, mask)['last_hidden_state'][:,:3], actual['last_hidden_state'][:,:3])
    torch.testing.assert_close(actual['last_token'][1], actual['last_hidden_state'][1,2])
    logits = model.hidden_state_to_token(actual['last_hidden_state'])
    loss = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,41), ids[:,1:].reshape(-1))
    loss.backward()
    assert model.gpt_layers[0].self_attention.query.weight.grad.abs().sum() > 0

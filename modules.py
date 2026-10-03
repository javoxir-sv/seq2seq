import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim


def generate_name(model, id_to_token, token_to_id, prefix=None, temperature=1, max_range=20, device="cpu", retn=False):
    model.eval()
    sos_id = token_to_id.get("<SOS>")
    eos_id = token_to_id.get("<EOS>")
    sequence_ids = [sos_id]

    if prefix is not None:
        for char in prefix:
            char = char.lower()
            sequence_ids.append(token_to_id[char])

    with torch.no_grad():
        for _ in range(max_range - len(sequence_ids)):
            x_tensor = torch.tensor([sequence_ids], dtype=torch.long).to(device)

            logits, _ = model(x_tensor)
            logits = logits[0, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_char_id = torch.multinomial(probs, num_samples=1).item()
            
            if next_char_id == eos_id:
                break

            sequence_ids.append(next_char_id)

    generated_name = "".join([id_to_token[idx] for idx in sequence_ids[1:]])
    if retn:
        return generated_name
    else:
        print(generated_name)


def translate_name(model, config, name=None, device="cpu", max_range=20, attention=False):
    model.eval()
    token_to_id = config['token_to_id']
    id_to_token = config['id_to_token']
    sos_id = token_to_id.get("<SOS>")
    eos_id = token_to_id.get("<EOS>")
    seq_len = config['seq_len']

    sequence_ids = [sos_id]
    if name is not None:
        for char in name:
            char = char.lower()
            sequence_ids.append(token_to_id[char])
    sequence_ids.append(eos_id)
    num_pads = [0] * (seq_len - len(sequence_ids))
    sequence_ids.extend(num_pads)
    sequence_ids = torch.tensor([sequence_ids], dtype=torch.long, device=device)

    translated_indices = []
    input_token = torch.tensor([[sos_id]], device=device)
    
    with torch.no_grad():
        _, (encoder_outputs, (hidden, cell)) = model.encoder(sequence_ids)
        for _ in range(max_range - len(translated_indices)):
            if attention:
                logits, (hidden, cell) = model.decoder(input_token, hidden, cell, encoder_outputs)
            else:
                logits, (hidden, cell) = model.decoder(input_token, hidden, cell)
            next_token = logits.argmax(dim=-1).item()
            if next_token == eos_id:
                break
                
            translated_indices.append(next_token)
            input_token = torch.tensor([[next_token]], device=device)
            
    generated_name = "".join([id_to_token.get(idx, "") for idx in translated_indices])
    return generated_name


def get_names(dataset, num, id_to_token):
    rng = np.random.default_rng()
    name_tokens = dataset.__getitem__(rng.integers(len(dataset), size=num))
    english = name_tokens[0].numpy()
    russian = name_tokens[1].numpy()
    
    en_names = []
    ru_names = []
    for i in range(5):
        en_name = "".join([id_to_token.get(idx) for idx in english[i]])
        en_names.append(en_name.split("<")[1].split(">")[1])
        ru_name = "".join([id_to_token.get(idx) for idx in russian[i]])
        ru_names.append(ru_name.split("<")[1].split(">")[1])
    return en_names, ru_names



# A function to get one batch of logits from Seq2SeqTranslator model
# The output of this function is already aligned.
def get_logits(model, dataset, device):
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in dataset:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch[:, :-1], y_batch[:, :-1])
            return logits, (x_batch[:, 1:], y_batch[:, 1:])



def perplexity(logits, targets, pad_idx=0):
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)

    ce_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_idx)
    ppe = torch.exp(ce_loss).item()
    return ppe


def plot_attention_matrix(attention_weights, source_tokens, target_tokens, epoch):
    if isinstance(attention_weights, list):
        stacked_attn = torch.cat(attention_weights, dim=2)
    else:
        stacked_attn = attention_weights
    attn_data = stacked_attn[0].detach().cpu().numpy() 
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Cross-Attention Matrices at Epoch {epoch}', fontsize=16)
    for i in range(3):
        head_data = attn_data[i]
        
        sns.heatmap(head_data, ax=axes[i], cmap='viridis', 
                    xticklabels=source_tokens, yticklabels=target_tokens)
        axes[i].set_title(f'Attention Head {i+1}')
        axes[i].set_xlabel('Source Sequence')
        axes[i].set_ylabel('Target Sequence (Decoder Input)')
        
    plt.tight_layout()
    plt.show()


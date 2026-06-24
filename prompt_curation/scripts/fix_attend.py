with open("step1_generate_and_attend.py") as f:
    c = f.read()

old = """    step0_atts = output.attentions[0]  # tuple of (batch,heads,1,input_len)
    att_stack  = torch.stack([a[0] for a in step0_atts], dim=0)  # (layers,heads,1,input_len)
    att_mean   = att_stack.mean(dim=(0, 1))[0]  # (input_len,)
    att_mean   = att_mean / (att_mean.sum() + 1e-8)
    att_weights = att_mean.cpu().float().numpy().tolist()"""

new = """    # step0: tuple of 32 layers, each (batch, heads, seq, seq)
    # seq = input_len + 1 (full causal self-attention)
    # Take last row of mean attention, slice to input_len only
    step0_atts = output.attentions[0]
    att_stack  = torch.stack([a[0] for a in step0_atts], dim=0)  # (layers,heads,seq,seq)
    att_mean   = att_stack.mean(dim=(0, 1))                        # (seq, seq)
    last_row   = att_mean[-1, :input_len]                          # (input_len,)
    last_row   = last_row / (last_row.sum() + 1e-8)
    att_weights = last_row.cpu().float().tolist()"""

assert old in c, "Pattern not found"
c = c.replace(old, new)
with open("step1_generate_and_attend.py", "w") as f:
    f.write(c)
print("Fixed")

import random
import torch


class Batch_provider():
    def __init__(self, chorales_dict, merged_chorales, block_size, batch_size, device):
        self.merged_chorales = merged_chorales
        self.chorales_dict = chorales_dict
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        n = int(0.8*len(merged_chorales))
        self.train_data = merged_chorales[:n]
        self.val_data = merged_chorales[n:]
        self.train_index = int(0.8*len(chorales_dict))
        self.val_index = int(0.2*len(chorales_dict))

    
    def get_chorale_batch(self, split):
        data = self.train_index if split == 'train' else self.val_index
        ix = random.randint(0, data)
        chorale = self.chorales_dict[ix]
        ix = torch.randint(len(chorale) - self.block_size, (self.batch_size, ))
        x = torch.stack([chorale[i:i+self.block_size] for i in ix]) 
        y = torch.stack([chorale[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
    
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size, ))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y


max_iters = 5000
eval_iters = 200

@torch.no_grad()
def estimate_loss(model, batch_provider, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batch_provider.get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(optimizer, model, batch_provider, eval_iters):
    for item in range(max_iters):
        if item % eval_iters == 0:
            losses = estimate_loss(model, batch_provider, eval_iters)
            print(f"step: {item}\t | train_loss: {losses['train']:.4f} | val_loss: {losses['val']:.4f}")
        # sample batch of data
        xb, yb = batch_provider.get_chorale_batch('train')

        # evaluate loss
        _, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())
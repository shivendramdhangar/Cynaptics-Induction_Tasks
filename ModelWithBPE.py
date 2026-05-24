import torch
from time import time
import torch.nn as nn
from torch.nn import functional as F
from dataloader import load_dataset
from BPEncoder import decode,encode,vocab
#hyperparamters
batch_size=64 #how many indeppendent sequences will we process in parallel
block_size=256#what is the maximum context length for predictions
max_iters=7500
eval_interval=500
learning_rate=1e-4
eval_iters=200
n_embd=192
dropout=0.2
n_head=4
n_layer=4
device='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
text=load_dataset(True)

#here are all unique characters that occur in this text
chars=sorted(list(set(text)))

start=time()
#create a mapping from characters to integers
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}

#encode=lambda s:[stoi[ch] for ch in s] #encoder takes a string, output a list of integers
#decode=lambda l: ''.join([itos[i] for i in l])#decoder: takes a lsit ofintegers, output a string
vocab_size=len(vocab)
#train test split
data=torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

#data loading
def get_batch(split):
    #generate a small batch of data of inputs of x and y
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size-1, (batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+1+block_size] for i in ix])
    x,y = x.to(device) , y.to(device)
    return x,y
print(time()-start)
@torch.no_grad()
def estimate_loss():
#  It evaluates model performance on:training data validation data, Without updating weights
#for one batch,it could be lucky and get less error that's why taking many random batches and averaging error
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item() #.item() converts tensor to float,you’re inserting a tensor into another tensor
        out[split]=losses.mean() #can create unnecessary computation graph links
    model.train() #train loss ↓ but val loss ↑
    return out
class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key=nn.Linear(n_embd, head_size, bias=False)
        self.query=nn.Linear(n_embd, head_size, bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        #compute attention scores("affinities")
        wei=q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 #(B,T,hs) @(B,hs,T) -> (B,T,T)
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) #(B,T,T)
        #what does masked_fill do? where mask==True, replace value
        #                          where mask==False, keep original
        #shape mismatich ,wei is (B,T,T) but ,mask is (T,T), pytorch broadcasts it to (B,T,T) same mask applied to every batch
        #why before softmax, softmax decides porbability distribution
        wei=F.softmax(wei, dim=-1) #(B,T,T)
        wei=self.dropout(wei) #dropout randomly sets some attention weights->0
        #prevents model from depending too much on specific tokens
        #with dropout: sometimes that token disappears ... find alternatives"
        #without dropout:always attend 100% to previous word, overfits, brittle behavior
        #perform the weighted aggregationn of the values
        v=self.value(x) #(B,T,C)
        out=wei @ v #(B,T,T) @(B,T,C)->(B,T,C)
        return out
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention is parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(head_size*num_heads, n_embd)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        out= torch.cat([h(x) for h in self.heads], dim=-1)
        out=self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module): #MLP, applied independently to each token
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_embd,4* n_embd), #increase capacity to learn complex patterns
                                nn.ReLU(), #adds non-linearity
                                nn.Linear(4*n_embd, n_embd),
                                nn.Dropout(dropout),)

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        #n_embd: embedding dimensionn, n_head: the number of heads we'd like
        super().__init__()
        head_size=n_embd//n_head
        self.sa=MultiHeadAttention(n_head, head_size)#tokens talk to each other
        self.ffwd=FeedForward(n_embd)#each token processes information deeply 
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)
        #Layer norm , normalizes eaxh token's feature vector, mean=0, var=1,across embedding dimension
        #Without layer norm :values explode or vanish, training become unstable,gradient become useless
        #stabilizes input before heavy operations,better gradient flow, More stable training
    def forward(self,x):
        x=x + self.sa(self.ln1(x)) #it contain learnable parameters like gamma(learnable scale), beta(learnable shift)
        #   ^-- residual connection , x = accumulated knowledge across layers
        x=x + self.ffwd(self.ln2(x)) #not fixed distribution but learnable distribution
        return x
        #Attention needs balanced values,stable dot prodcuts(q.k), no extreme magnitued::prefers:controlled, symmetric distribution
        #FeedForward need strong activations, non-linear separations(ReLU), some neurons high, some low::prefers:more spread/expressive distribution
        #even though both start at mean=0,var=1, ln(x)->tuned for attention, ln2(x)-->tuned for FFN
#super simple biagram model
#class BiagramLanguageModel(nn.Module):
print(time()-start)
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd) #n_embd=no of embedding dimension #
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        #self.sa_heads=MultiHeadAttention(4, n_embd//4) #i.e. 4 heads of 8-dimensional self-attention
        #self.ffwd=FeedForward(n_embd)
        #self.blocks=nn.Sequential(Block(n_embd,n_head=4),
        #                          Block(n_embd,n_head=4),
        #                          Block(n_embd,n_head=4),
        #                         nn.LayerNorm(n_embd))
        self.blocks=nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd) #final layer norm
        self.lm_head=nn.Linear(n_embd, vocab_size) #token embedding to logits
        
        self.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self,idx,targets=None):
        B,T=idx.shape
        #idx and targets are both (B,T) tensor of integers
        tok_emb=self.token_embedding_table(idx) #(B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x=tok_emb + pos_emb #(B,T,C)
        #x=self.sa_heads(x) #apply one head of self-attention. (B,T,C)
        #x=self.ffwd(x) #(B,T,C)
        x=self.blocks(x) #(B,T,C)
        x=self.ln_f(x) #(B,T,C)
        logits= self.lm_head(x)  #(B,T,vocab_size)

        if targets is None:
            loss=None
        else:
            B,T,C= logits.shape
            logits_flat=logits.view(B*T,C)
            targets_flat=targets.view(B*T)
            loss=F.cross_entropy(logits_flat,targets_flat) #logits:[2.0, 1.0, 0.1]target:0,softmax → [0.7, 0.2, 0.1],loss = -log(0.7)
        
        return logits, loss #Returned logits = (B, T, C) Used flattened version only for loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond=idx[:, -block_size:]
        #get the predictions
            logits,loss=self(idx_cond)
            #focus only on the last time step
            logits=logits[:,-1,:] #becomes (B,C)
            #apply softmax to get probabilities
            probs=F.softmax(logits/0.8, dim=-1) #(B,C) -1 means last dimension ([C numbers] → convert into probabilities)
            #sample from the distribution #why multinomial, adds controlled randomness, creative,diverese,human-like
            idx_next=torch.multinomial(probs, num_samples=1)#(B,1) #why not argmac=x, always picks highest probability token ,Problem: deterministic,boring output, repetitive text
            #append samples index to the running sequence
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx
    
#model=BiagramLanguageModel()
model=GPTLanguageModel()

m=model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
 #create a PyTorch optimizer
optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-2)
# Model makes predictionLoss is computed Gradients are calculated (backprop) Optimizer updates weights

print(time()-start)
for iter in range(max_iters):
    if iter>5000:
        lr=5e-5
    #every once in a while evaluate the loss on train and val loss
    if iter% eval_interval==0 or iter==max_iters-1:
        losses=estimate_loss()
        print(f"step{iter}:train loss{losses['train']:.4f}, val losss {losses['val']:.4f}")
    
    #sample a batch of data
    xb, yb=get_batch('train')
    logits,loss=model(xb,yb) #prediction
    optimizer.zero_grad(set_to_none=True) #reset gradients
    loss.backward() #computes gradients
    optimizer.step() #update weights

print(time()-start)
#generate from the model
input_text="""Lord Manderly:
Nay, we shall not bend the knee to southern steel,
Nor crown a boy who claims the dragon's throne.
The wolves of Winterfell alone command us here
Lord Umber:
We fight, or perish ere we yield our pride!
All Lords together:
No king but the King in the North! Stark! Stark!
The King in the North!"""
context=torch.tensor(encode(input_text),dtype=torch.long , device=device)
context=context.unsqueeze(0)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))

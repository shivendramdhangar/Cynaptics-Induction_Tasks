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

# Cynaptics-Induction_Tasks
GPT Language Model from Scratch (Tiny Shakespeare)
Overview

This project implements a decoder-only Transformer (GPT-style language model) from scratch using PyTorch. The model is trained on the Tiny Shakespeare dataset to generate Shakespeare-like text.

The goal of this project is to demonstrate a deep understanding of:

    Transformer architecture
    Self-attention mechanism
    Language modeling pipeline
    Training and generation process

Model Architecture

The model follows the standard GPT pipeline:

    Token Embeddings

    Positional Embeddings

    Stacked Transformer Blocks
        Multi-Head Self Attention
        Feed Forward Network (MLP)
        Layer Normalization + Residual Connections

    Final LayerNorm

    Linear Projection to Vocabulary

Key Concepts Implemented

    Bigram → Transformer transition
    Causal masking (no future information leakage)
    Multi-head attention
    Residual connections
    Layer normalization (Pre-Norm)
    Dropout regularization
    Autoregressive text generation

Training Details

    Dataset: Tiny Shakespeare
    Batch Size: 64
    Block Size (context length): 256
    Embedding Size: 192
    Heads: 4
    Layers: 4
    Dropout: 0.2
    Learning Rate: 1e-4
    Training Steps: 7500

Loss Progress
Step 	Train Loss 	Val Loss
0 	6.25 	6.25
2000 	3.15 	3.60
3000 	2.89 	3.42
4000 	2.76 	3.37
7600 	2.47 	3.29

Model begins to overfit after ~7500 steps
Given Input:


Lord Manderly:
Nay, we shall not bend the knee to southern steel,
Nor crown a boy who claims the dragon's throne.
The wolves of Winterfell alone command us here
Lord Umber:
We fight, or perish ere we yield our pride!
All Lords together:
No king but the King in the North! Stark! Stark!

Sample Generated Output(BPE used)

The King in the North! I must be done?

BENVOLIO:
Ah'st thou weary me from the Follower,
Thou art about'st of his hand:
And this have a carr'd, though childly sent merve.

HERMIONE:
The'er I will be often'd
I front this chamber'd by the day
Her lamentation of her ground.
Take the stuffixile too mistress it in
With growth showed to prove the breast: who wakes this wretchs
Again already to Brittan's son,
And am each on your highness follible. Alas, let us rest,
We thee: I pray him ale.

LUCIO:
I'll never to see hopes Paris, yet could you extedious
Had I could be hang in this feditt;
But Camillo, fret married within pure should have prosperous;
To know mine honourable; she ses in set itedition about your souls     
I made no fortune?
She were not so disgraced, and go with thy love.

BUCKINGHAM:
You will not buy my heart an ever to Lady
Than the clocks of my daughter's hand.
What good succeed are commitive to stand
And direction: all his fleety out of my blood
Betwixister, say you think.

BUCKINGHAM:
Out up my somerces to do pleasest wear
And you as I pray you?

##Output without BPE (Given Input was tensor([[0]]

"Newly and device
To crave this light percupture follow temper,
A throne's hoicing, his voice, a king the north,
A seal'd under an oen, a villain,
Anst the sacrifular's shadow, we state,
Which are first worn in the midrict
Of which neighbours sided to their crown,
And bear it ransomen'd. They have money at no,
Which made it stands, to doom down! Strike so, and
A rest that modest are one. Thou villain,
I never shall do well truth away to him:
If I have said this line evils to make
The lady's dark"



ASK 2: SUPERVISED FINE-TUNING OF GPT-2 ON ALPACA DATASET
Overview

This project demonstrates Supervised Fine-Tuning (SFT) of GPT-2 on the Alpaca dataset to improve instruction-following capability.

The base GPT-2 model is not designed for instruction-based tasks. After fine-tuning, it learns to generate structured responses to prompts like:

I have uploaded SFT1.py and SFT2.py , I could only run SFT1 (but only 0.5 epoch took three hours), SFT2 is some changes in SFT1 ( calculates loss only on response and there is added eos_token at the end of formatted sample so model could learn when to stop)
What Was Done

    Loaded GPT-2 (openai-community/gpt2)
    Used Alpaca dataset (~52K samples)
    Formatted data into instruction-response pairs
    Applied masked loss (loss computed only on response)
    Tokenized with GPT-2 tokenizer
    Trained using HuggingFace Trainer

Model Details

    Model: GPT-2 (124M parameters)

    Architecture:
        Layers: 12
        Heads: 12
        Embedding size: 768

Hyperparameters

    Batch size: 2(GPU constraint)
    Epochs: ~0.5 (training stopped early due to time constraints)
    Learning rate: 5e-5 (scheduled learning rate)
    Sequence length: 512
    FP16: Enabled

Limitations:

    Training incomplete (~50% epoch) due to time.

Note

Training was stopped at ~50% due to time constraints. Despite this, the model shows noticeable improvement over base GPT-2 in instruction-following tasks.

Outputs:

##Instruction:

Explain Photosynthesis simply

##Response

Photosynthetic organisms use sunlight to create energy, whereas non-renewable organisms require water to make oxygen and carbon dioxide. Plants can use the sun's energy for their growth in order of appearance or coloration; plants are unable access this energy from other sources such as sunlight, and cannot use it to produce food. Plants also have difficulty using the sun's light due its wavelength, so they must rely on sunlight to provide energy.

Instrucion

What is Machine Learning

Response:

Machine learning algorithms are used to generate and analyze large amounts of data quickly, accurately or efficiently in various tasks such as text classification (SVD), sentiment analysis/emotion recognition systems for speech detection applications like facial recognition software; natural language processing technologies like AI-driven machine translation; and image recognition algorithms related from text generation through inference techniques including convolutional neural networks (CNNs).        

Instruction:

How to fine tune a model

Response:

A fine tuning can be done by first adjusting its parameters according to the following criteria - the model should be optimized for natural language processing (NLP) tasks, such as sentiment analysis and classification algorithms; then optimizing this model's performance accordingly. Finally distancing any outliers from results will help to reduce the accuracy of data.

Author

Shivendram Dhangar

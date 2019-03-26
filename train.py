import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Create model directory
    if not os.path.exists('models/'):
        os.makedirs('models/')
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader('data/resized2014', 'data/annotations/captions_train2014.json', vocab, 
                             transform, 128,
                             shuffle=True, num_workers=2) 

    # Build the models
    encoder = EncoderCNN(256).to(device)
    decoder = DecoderRNN(256, 512, len(vocab), 1).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(5):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, 5, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % 1000 == 0:
                torch.save(decoder, os.path.join(
                    'models/', 'decoder-{}-{}.pkl'.format(epoch+1, i+1)))
                torch.save(encoder, os.path.join(
                    'models/', 'encoder-{}-{}.pkl'.format(epoch+1, i+1)))


if __name__ == '__main__':

    main()



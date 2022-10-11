from cProfile import label
from itertools import dropwhile
from pickle import TRUE
from sched import scheduler
from typing import List
from unicodedata import name
from torch.nn import LSTM, MultiheadAttention, Linear
from torch import tensor, rand, zeros, matmul, permute, log, lgamma, cat, empty
from torch import nn
from torch import optim
from tqdm import trange
import torch.nn.functional as F
import numpy as np
import torch

from pyro.distributions import Dirichlet
from pyro.contrib.forecast import eval_crps 

import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

from plot_utils import plot_grad_flow

class Dirichlet_exp(object):
    def __init__(self, alpha):
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                           np.multiply.reduce([gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * np.multiply.reduce([xx ** (aa - 1)
                                               for (xx, aa)in zip(x, self._alpha)])

def get_loss(
    output:torch.tensor, 
    target:torch.tensor, 
    epsilon:float = 1e-02, 
    threshold: float = 1e-08, 
    complex_modification : bool = True
    ) -> torch.tensor:

    """ Compute the negative log-likeihood for all observations on forecasting horizon 

    - if complex_modification is False, the epsilon is forced to be 1e-08
    for numerical stability, and this modification is applied universally 

    - if complex_modification is True, we scan the target tensor, if there are anmoloy 
    points found (hit the threshold), we provide a self-coherent modification based on 
    epsilon

    """
    drichilet = Dirichlet(concentration=output)

    row_indexs = (target >= 1-threshold).nonzero()[:,0]
    col_indexs = (target >= 1-threshold).nonzero()[:,1]

    if complex_modification:

        for i in range(len(row_indexs)): 
            target[row_indexs[i],col_indexs[i]] -= epsilon

            target[row_indexs[i],0:col_indexs[i]] += epsilon/2
            target[row_indexs[i],col_indexs[i]+1:] += epsilon/2

        row_indexs = (target <= threshold).nonzero()[:,0]
        col_indexs = (target <= threshold).nonzero()[:,1]

        for i in range(len(row_indexs)): 
            target[row_indexs[i],col_indexs[i]] += epsilon

            target[row_indexs[i],0:col_indexs[i]] -= epsilon/2
            target[row_indexs[i],col_indexs[i]+1:] -= epsilon/2
    else: 
        epsilon = 1e-08
        target += epsilon

    loss = drichilet.log_prob(target)
    #print(loss)
    loss_sum = loss.sum()

    return -loss_sum

class batch_norm(nn.Module): 

    def __init__(self, num_features) -> None:
        super(batch_norm, self).__init__()

        self.num_features = num_features
        self.batch_norm_layer = nn.BatchNorm1d(self.num_features)

    def forward(self, x): 

        return self.batch_norm_layer(x)


class hts_embedding(nn.Module):

    """
    The embedding class for the time series
    input:
    - num_embeeddings and the dimension of embeddings
    - x: the input that represents the index of the time series, starting from zero

    """

    def __init__(
        self,
        num_embedd,
        embedd_dim,
    ) -> None:
        super(hts_embedding, self).__init__()

        self.num_embedd = num_embedd
        self.embedd_dim = embedd_dim
        self.embedd_layer = nn.Embedding(
            self.num_embedd, self.embedd_dim, max_norm=True
        )

    def forward(self, x):

        output = self.embedd_layer(x)

        return output


# unidirectional implementation of the LSTM Seq-2-Seq Network
class encoder_lstm(nn.Module):

    """Encodes time-series sequence"""

    def __init__(
        self, 
        ts_embedding_dim, 
        lstm_input_dim, 
        lstm_hidden_dim, 
        lstm_num_layer, 
        batch_first=True, 
    ) -> None:

        super(encoder_lstm, self).__init__()

        """
        : param lstm_input_dim:     the number of features in the input X
        : param lstm_hidden_dim:    the number of features in the hidden state h
        : param lstm_num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        : batch_first:          True --> first input would be the number of sequences/observations
        """

        self.ts_embedding_dim = ts_embedding_dim
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layer = lstm_num_layer
        self.batch_first = batch_first
        #self.dropout = dropout
        self.linear = Linear(ts_embedding_dim, self.lstm_input_dim)

        self.lstm = LSTM(
            self.lstm_input_dim,
            self.lstm_hidden_dim,
            self.lstm_num_layer,
            batch_first=self.batch_first,
            #dropout = dropout,
        )

        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        """
        : param x :                    input of shape (# in batch, seq_len, lstm_input_dim)

        : return lstm_out,:            lstm_out gives all the hidden states in the sequence;

        : return self.cache            gives the hidden state and cell state for the last
        :                              element in the sequence
        """

        lstm_input = F.relu(self.linear(x))
        encoder_ouptut, endoer_hidden_output = self.lstm(lstm_input)

        return encoder_ouptut, endoer_hidden_output


class decoder_lstm(nn.Module):

    "Decodes time-series sequence"

    def __init__(
        self,
        lstm_input_dim,
        lstm_hidden_dim,
        lstm_num_layer,
        lstm_ouptut_dim,
        batch_first=True,
    ) -> None:
        super(decoder_lstm, self).__init__()

        """
        : param self.lstm_input_dim:     the number of features in the input X
        : param lstm_hidden_dim:    the number of features in the hidden state h
        : param lstm_num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        : batch_first:          True --> first input would be the number of sequences/observations
        """

        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layer = lstm_num_layer
        self.lstm_output_dim = lstm_ouptut_dim
        self.batch_first = batch_first
        self.lstm = LSTM(
            self.lstm_input_dim,
            self.lstm_hidden_dim,
            self.lstm_num_layer,
            batch_first=self.batch_first,
        )
        self.linear_1 = Linear(self.lstm_hidden_dim, self.lstm_input_dim)
        self.linear_2 = Linear(self.lstm_input_dim, self.lstm_output_dim)

    def forward(self, x, endoer_hidden_output):

        """
    
        x: input at the last timestamp  input of shape should 2D (batch_size, input_size)

        encoder_output: cache of the encoder output
            hn: (D time lstm_num_layers, # nathces, lstm_hidden_dim))
            cn: (D time lstm_num_layers, # nathces, lstm_hidden_dim))
            D = 2 if bi-directional, 1 if unidirectional
        """ 

        ### commented the line due to the fact that we are running a batched-decoder LSTM
        # if self.batch_first:
        #     x = x.unsqueeze(1)

        # else:
        #     x = x.unsqueeze(0)

        decoder_output, decodoer_hidden_output = self.lstm(x, endoer_hidden_output)
        decoder_output = self.linear_1(decoder_output)

        ### commented the line due to the fact that we are running a batched-decoder LSTM
        # if self.batch_first:
        #     decoder_lstm_output = decoder_lstm_output.squeeze(1)
        # else:
        #     decoder_lstm_output = decoder_lstm_output.squeeze(0)
        layer_output = F.relu(self.linear_2(decoder_output))

        return layer_output, decoder_output, decodoer_hidden_output


class mha_with_residual(nn.Module):
    def __init__(
        self,
        mha_embedd_dim,
        num_head,
        mha_output_dim,
        activation=nn.ReLU(),
        batch_first=False,
    ) -> None:
        super().__init__()

        assert (
            mha_embedd_dim % num_head == 0
        )  # Embedding dimension must be 0 module number of heads

        self.mha_embedd_dim = mha_embedd_dim
        self.num_head = num_head
        self.batch_first = batch_first
        self.mha_output_dim = mha_output_dim
        self.activation = activation
        self.mha = MultiheadAttention(
            self.mha_embedd_dim, 
            self.num_head, 
            batch_first=self.batch_first
        )
        self.linear = Linear(self.mha_output_dim, self.mha_output_dim)
        self.batch_norm_layer = nn.BatchNorm1d(self.mha_embedd_dim)
        #self.layer_norm_layer = nn.LayerNorm(self.mha_embedd_dim)

    def forward(self, x:torch.tensor,):

        """forward prop for one Multi-headed Attention layer

        Returns:
            x: (torhch.tensor) Dimension: (
                F : -> forecasting horizon 
                C : -> number of child in this family 
                Dim: -> ouptut dimension of the MHA/LSTM 
                    accroding to the paper is 64 for M5 
            )
        """
        # BATCH FIRST  = TRUE for MHA 
        values, atten_wieghts = self.mha(x, x, x)
        values = self.linear(values)
        values = values + x
        values = self.activation(values)
        #values = self.layer_norm_layer(values)
        values = self.batch_norm_layer(values.permute(0,2,1))
        values = values.permute(0,2,1) 

        return values, atten_wieghts


class output(nn.Module):
    def __init__(self, mha_output_dim, model_output_dim) -> None:
        super().__init__()

        self.mha_output_dim = mha_output_dim
        self.model_output_dim = model_output_dim
        self.linear = nn.Linear(self.mha_output_dim, self.model_output_dim)

    def forward(self, x):

        output = self.linear(x)
        output = torch.abs(output)
        #output = torch.clamp(output, max=50)

        return output


class ProportionModel(nn.Module):
    def __init__(
        self,
        target_len : int,
        num_hts_embedd,
        hts_embedd_dim,  # ts embedding hyper pars
        lstm_input_dim,
        lstm_hidden_dim,
        lstm_num_layer,
        lstm_output_dim,  # lstm hyper pars
        mha_embedd_dim,
        num_head,
        num_attention_layer,  # mha hyper pars
        mha_output_dim,
        residual_output_dim,  # skip connection hyper pars
        model_ouput_dim,  # output later hyper pars
        mha_activation = nn.ReLU(),
        covariate_dim = 0,
    ) -> None:
        super(ProportionModel, self).__init__()

        """
        lstm_input_dim: input dimensiotn of the LSTM encoder 
        lstm_hidden_dim: fixed hidden dimension of the LSTM encoder and decoder 
        lstm_num_layer: number of the layers in the 
        
        
        """

        assert model_ouput_dim == 1
        self.covariate_dim = covariate_dim

        self.num_hts_embedd = num_hts_embedd
        self.hts_embedd_dim = hts_embedd_dim

        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layer = lstm_num_layer
        self.lstm_output_dim = lstm_output_dim

        self.mha_embedd_dim = mha_embedd_dim
        self.num_head = num_head
        self.num_attention_layer = num_attention_layer
        self.mha_output_dim = mha_output_dim
        self.residual_output_dim = residual_output_dim
        self.model_ouput_dim = model_ouput_dim

        self.batch_norm_layer = batch_norm(self.lstm_input_dim)
        self.embedd_layer = hts_embedding(self.num_hts_embedd, self.hts_embedd_dim)
        self.encoder_lstm = encoder_lstm(
            self.hts_embedd_dim + self.covariate_dim + 2,
            self.lstm_input_dim,
            self.lstm_hidden_dim,
            self.lstm_num_layer,
            batch_first=True,
        )
        self.decoder_lstm = decoder_lstm(
            self.lstm_input_dim,
            self.lstm_hidden_dim,
            self.lstm_num_layer,
            self.lstm_output_dim,
            batch_first=True,
        )
        self.mha_with_residual = mha_with_residual(
            self.mha_embedd_dim,
            self.num_head,
            self.mha_output_dim,
            activation= mha_activation,
            batch_first=True,
        )
        self.output = output(self.mha_output_dim, self.model_ouput_dim)
        self.target_len = target_len


    def forward(
        self,
        input :torch.tensor,
    ): 
        """
        implement the forward propogation for each single observation 

        input: (C, H, dim), embedding dim always the last layer 
        """
        # if not training: 
        #     self.eval()

        # empty tensor to hold decoder ouputs 
        decoder_ouputs = empty(
            self.target_len, 
            input.shape[0],
            self.lstm_output_dim,
        )

        # split the features
        embedding_input = (input[:, 0, -1]).long() 
        feature_input = input[:, :, :-1] 

        # forward prop -- embedding layer 
        embedd_vector = self.embedd_layer(embedding_input) 
        embedd_vector = torch.unsqueeze(embedd_vector, axis=1)
        embedd_vector = embedd_vector.repeat(1,input.shape[-2],1) 
        
        # concate feaute and embeddings 
        input = cat((feature_input , embedd_vector), -1)  

        # forward prop -- batch norm 
        input = input.permute(0,2,1)
        input = self.batch_norm_layer(input)
        input = input.permute(0,2,1) 

        # forward prop - LSTM encoder 
        encoder_ouput, endoer_hidden_output = self.encoder_lstm.forward(input)

        # last timestamp from the input
        decoder_input = torch.unsqueeze((input[:, -1, :]), dim = 1)
        decoder_cache = endoer_hidden_output

        # forward prop - LSTM decoder 
        for t in range(self.target_len):
            (layer_output, decoder_output,decodoer_hidden_output,) = self.decoder_lstm.forward(decoder_input, decoder_cache)
            
            decoder_ouputs[t, :, :] = torch.squeeze(layer_output, dim = 1)
                
            decoder_input = decoder_output
            decoder_cache = decodoer_hidden_output

        # forward prop - multi-headed attention 
        # decoder outputs (F, C, dim)
        attention_inputs = decoder_ouputs
        for i in range(self.num_attention_layer):

            value, attention_weights = self.mha_with_residual.forward(attention_inputs)
            attention_inputs = value 

            ##### -------- adding the batch norm to make sure the gradient is under control ------ #### 

        # forward prop - linear output 
        output = self.output(value)  # L, C, 1

        # collapse the last dimension 
        output = torch.squeeze(output, dim=-1) 

        return output, decoder_ouputs, value 

    def evaluate_distribution_crps(self, input: torch.tensor, target: torch.tensor, n_samples:int = 200) -> float: 

        self.eval()
        output, _, _  = self.forward(input=input)
        target = torch.squeeze(target, dim=-1)
        target = torch.permute(target, (1,0))
        
        predictive_distribution = Dirichlet(output)
        samples = predictive_distribution.sample([n_samples])
        crps = eval_crps(samples, target)
        return crps


def train_model(
    model:ProportionModel,  
    input_tensor: torch.tensor, 
    target_tensor: torch.tensor, 
    n_epochs :int, 
    target_len : int, 
    batch_size :int, 
    learning_rate : float,
    PATH = "model.pt",
    clip : bool = False,
    tracing : bool = True, 
    val_input_tensor : torch.tensor = None,
    val_target_tensor: torch.tensor = None, 
    epsilon: float = 1e-05,
    threshold: float = 0.95,
    complex_modification : bool = True,
    ):
    """
    All implementation is based on Batch = False
    input_tensor: 4D torch tensor of shape b, H, C, input_dim
    target_tensor: 4D torch tensor of shape b, F, C, 1
        b: time-batched input
        C: number of the children
        H: history data points
        F: future data points
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ExponentialLR(
    #     optimizer=optimizer,
    #     gamma=0.5,
    # )

    # try: 
    #     checkpoint = torch.load(PATH) 

    #     if learning_rate == checkpoint['lr']: 

    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 

    #         iter_start = checkpoint['epoch']
    #         batch_start = checkpoint['batch']

    #     else: 
    #         iter_start = 0 
    #         batch_start = 0 

    # except FileNotFoundError:

    iter_start = 0 
    batch_start = 0 
        #pass 

    print(f"Trainign starting from iteration {iter_start}, batch {batch_start}")

    number_observations = input_tensor.shape[0]

    n_complete_batches = number_observations//batch_size 
    n_batches = n_complete_batches + 1

    print(f"There are {n_batches} batches per iteration")

    with trange(iter_start, n_epochs) as tr:
        iter_losses = []
        for epoch in tr:

            average_batch_losses = []
            crpss = []

            print(f'Trainign for Iteration: {epoch} starts')
            
            for b in range(batch_start, n_batches):

                print(f'Training for Iteration: {epoch} Batch {b} starts')

                #print(f'Iteration: {it} - Batch: {b}')
                no_child = input_tensor.shape[-3]

                ### select the dataset according to the batch size
                if b <= n_complete_batches-1:
                    input_batch = input_tensor[b*batch_size : (b+1) * batch_size]
                    target_batch = target_tensor[b*batch_size : (b+1) * batch_size]
                else: 
                    input_batch = input_tensor[b*batch_size:]
                    target_batch = target_tensor[b*batch_size:]
                    print(input_batch.shape)
                    print(target_batch.shape)

                decoder_batch_ouputs = zeros(input_batch.shape[0], target_len, no_child, model.lstm_output_dim,)
                attention_batch_outputs = zeros(input_batch.shape[0], target_len, no_child, model.mha_output_dim,)
                model_batch_outputs = zeros(input_batch.shape[0], target_len, no_child, model.model_ouput_dim,)

                batch_loss = 0.
                batch_loss_no_grad = 0.
                "---- ZERO GRAD ----"
                model.zero_grad()
                optimizer.zero_grad()
                # pass in each observation for forward propogation
                for batch_index in range(input_batch.shape[0]):
                
                    input = input_batch[batch_index, :, :, :]
                    target = target_batch[batch_index, :, :, :] 
                    #print(input.shape)

                    output, decoder_ouputs, value = model.forward(input)
                    decoder_batch_ouputs[batch_index] = decoder_ouputs 
                    attention_batch_outputs[batch_index] = value
                    model_batch_outputs[batch_index] = output.unsqueeze(dim=-1)
                    target = torch.squeeze(target, dim=-1)
                    target = torch.permute(target, (1,0))

                    loss = get_loss(
                        output, 
                        target, 
                        epsilon = epsilon, 
                        threshold=threshold,
                        complex_modification = complex_modification
                    )

                    if b == 469: 
                        print(loss)
                        print(output)
                        print(target)

                    # reduction method defulat to sum, instead of mean 
                    batch_loss += loss
                    batch_loss_no_grad += loss.item()

                "---- BackProp ----"
                if  b == 469: 
                    print(batch_loss)
                average_batch_loss = batch_loss/input_batch.shape[0]
                average_batch_loss.backward() 
                #batch_loss.backward()
                if clip: 
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, norm_type=2)
                optimizer.step()

                average_batch_loss_no_grad = batch_loss_no_grad/input_batch.shape[0]
                if average_batch_loss_no_grad >= 0: 
                    print(f'Loss exploded to {average_batch_loss_no_grad} at {b+1}')
                    average_batch_loss_no_grad = 0

                average_batch_losses.append(average_batch_loss_no_grad) 

                "---- EVAL ----"
                ## doing eval might destroy the model 
                crps = model.evaluate_distribution_crps(val_input_tensor, val_target_tensor, n_samples=100)
                model.train()
                crpss.append(crps)


                "---- Tracing  ----"
                if tracing:

                    plt.figure(1, figsize=(10,5))
                    plt.subplot(211)
                    plt.plot(
                        list(range(1,b+2)),
                        (average_batch_losses),
                        label = 'train'
                    )
                    plt.ylabel('Loss')
                    plt.grid()
                    
                    plt.figure(2, figsize=(10,5))
                    ave_grads = []
                    layers = []
                    for n, p in model.named_parameters():
                        if(p.requires_grad) and ("bias" not in n):
                            layers.append(n)
                            ave_grads.append(p.grad.abs().mean())
                    
                    plt.plot(ave_grads, alpha=0.3, color="b")
                    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
                    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
                    plt.xlim(xmin=0, xmax=len(ave_grads))
                    plt.xlabel("Layers")
                    plt.ylabel("average gradient")
                    plt.title("Gradient flow")
                    plt.grid(True)

                    plt.figure(3, figsize=(10,5))
                    plt.plot(
                        list(range(1,b+2)),
                        (crpss),
                        label = 'valid',
                    )
                    plt.ylabel('crps')
                    plt.grid()
                    
                if b%20 == 0 and b!= 0: 
                #   print(f"The loss for iteration {it} batch {b} is {batch_loss_no_grad/batch_size}")
                    "---- Saving ----- "
                    plt.show()
            
                    # torch.save(
                    #     {   'lr' : learning_rate,
                    #         'epoch': it,
                    #         'batch': b,
                    #         'model_state_dict': model.state_dict(),
                    #         'optimizer_state_dict': optimizer.state_dict(),
                    #         'loss': batch_losses,
                    #     }, 
                    #     PATH
                    # )
                "--- Validation ---- "

            #print(f"Training for iteration {it} is completed")
            iter_loss = sum(average_batch_losses)/n_batches 
            # learning rate decay
            #scheduler.step()
            #print(f"The total loss for iteration {it} is {iter_loss}")
            iter_losses.append(iter_loss)
            tr.set_postfix(loss="{0:.3f}".format(iter_loss))
    
    return model, iter_losses



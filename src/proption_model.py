from tabnanny import check
from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
from nbformat import write
from torch.utils.tensorboard import SummaryWriter 
from torch.nn import LSTM, MultiheadAttention, Linear
from torch import tensor, rand, zeros, matmul, permute, log, lgamma, cat, empty
from torch import nn
from torch import optim
from tqdm import trange
import torch.nn.functional as F
from dirichlet import * 
import numpy as np
import torch

writer = SummaryWriter()
torch.autograd.set_detect_anomaly(True)

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

def get_loss(output, target, epsilon):

    drichilet = Dirichlet(concentration=output)
    # output 
    # print(f"the batch shape is {output.shape[:-1]}")
    # print(f"the event shape is {output.shape[-1:]}")
    # print(drichilet)

    loss = drichilet.log_prob(target, epsilon)
    loss_sum = loss.sum(-1)

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
        self, lstm_input_dim, lstm_hidden_dim, lstm_num_layer, batch_first=False
    ) -> None:

        super(encoder_lstm, self).__init__()

        """
        : param lstm_input_dim:     the number of features in the input X
        : param lstm_hidden_dim:    the number of features in the hidden state h
        : param lstm_num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        : batch_first:          True --> first input would be the number of sequences/observations
        """

        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layer = lstm_num_layer
        self.batch_first = batch_first

        self.lstm = LSTM(
            self.lstm_input_dim,
            self.lstm_hidden_dim,
            self.lstm_num_layer,
            batch_first=self.batch_first,
        )

    def init_hidden(self, batch_size):

        """
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        """

        return (
            torch.randn(1 * self.lstm_num_layer, batch_size, self.lstm_hidden_dim),
            torch.randn(1 * self.lstm_num_layer, batch_size, self.lstm_hidden_dim),
        )

    def forward(self, x, ):

        """
        : param x :                    input of shape (# in batch, seq_len, lstm_input_dim)

        : return lstm_out,:            lstm_out gives all the hidden states in the sequence;

        : return self.cache            gives the hidden state and cell state for the last
        :                              element in the sequence
        """

        encoder_ouptut, (self.hn, self.cn) = self.lstm(x,)
        self.cache = (self.hn, self.cn)

        return encoder_ouptut, self.cache


class decoder_lstm(nn.Module):

    "Decodes time-series sequence"

    def __init__(
        self,
        lstm_input_dim,
        lstm_hidden_dim,
        lstm_num_layer,
        lstm_ouptut_dim,
        batch_first=False,
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

    def forward(self, x, encoder_cache):

        """

        x: input at the last timestamp  input of shape should 2D (batch_size, input_size)

        encoder_output: cache of the encoder output
            hn: (D time lstm_num_layers, # nathces, lstm_hidden_dim))
            cn: (D time lstm_num_layers, # nathces, lstm_hidden_dim))
            D = 2 if bi-directional, 1 if unidirectional
        """
        (hn, cn) = encoder_cache

        ### commented the line due to the fact that we are running a batched-decoder LSTM
        # if self.batch_first:
        #     x = x.unsqueeze(1)

        # else:
        #     x = x.unsqueeze(0)

        decoder_output, (self.hn, self.cn) = self.lstm(x, (hn, cn))
        self.cache = (self.hn, self.cn)
        decoder_output = self.linear_1(decoder_output)

        ### commented the line due to the fact that we are running a batched-decoder LSTM
        # if self.batch_first:
        #     decoder_lstm_output = decoder_lstm_output.squeeze(1)
        # else:
        #     decoder_lstm_output = decoder_lstm_output.squeeze(0)

        layer_output = self.linear_2(decoder_output)

        return layer_output, decoder_output, self.cache


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
        )  # Embedding dimension must be 0 modulo number of heads

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

    def forward(self, x):

        values, atten_wieghts = self.mha(x, x, x)
        values = self.linear(values)
        values = values + x
        values = self.activation(values)
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
        output = torch.exp(output)
        #output = torch.clamp(output, max=50)

        return output


class proportion_model(nn.Module):
    def __init__(
        self,
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
    ) -> None:
        super(proportion_model, self).__init__()

        """
        lstm_input_dim: input dimensiotn of the LSTM encoder 
        lstm_hidden_dim: fixed hidden dimension of the LSTM encoder and decoder 
        lstm_num_layer: number of the layers in the 
        
        
        """

        assert model_ouput_dim == 1
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
            activation=nn.ReLU(),
            batch_first=True,
        )
        self.output = output(self.mha_output_dim, self.model_ouput_dim)


    def forward(
        self,
        target_len,
        input :torch.tensor,
    ): 
        """
        implement the forward propogation for each single observation 

        input: (C, H, dim), embedding dim always the last layer 
        """

        # empty tensor to hold decoder ouputs 
        decoder_ouputs = empty(
            target_len, 
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
        encoder_ouput, ( encoder_hn, encoder_cn,) = self.encoder_lstm.forward(input) 
        encoder_cache = (encoder_hn, encoder_cn) 

        # last timestamp from the input
        decoder_input = torch.unsqueeze((input[:, -1, :]), dim = 1)
        decoder_cache = encoder_cache

        # forward prop - LSTM decoder 
        for t in range(target_len):
            (layer_output, decoder_output,(decoder_hn, decoder_cn),) = self.decoder_lstm.forward(decoder_input, decoder_cache)

            decoder_ouputs[t, :, :] = torch.squeeze(layer_output, dim = 1)

            decoder_input = decoder_output
            decoder_cache = (decoder_hn, decoder_cn)

        # forward prop - multi-headed attention 
        # decoder outputs (F, C, dim)
        attention_inputs = decoder_ouputs
        for i in range(self.num_attention_layer):
            value, attention_weights = self.mha_with_residual.forward(attention_inputs)
            attention_inputs = value 

            ##### -------- adding the batch norm to make sure the gradient is under control ------ #### 

        # forward prop - linear output 
        output = self.output(value)  # L, C, 1

        return output, decoder_ouputs, value 

def train_model(
    model :proportion_model, 
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

    try: 
        checkpoint = torch.load(PATH) 

        if learning_rate == checkpoint['lr']: 

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 

            iter_start = checkpoint['epoch']
            batch_start = checkpoint['batch']

        else: 
            iter_start = 0 
            batch_start = 0 

    except FileNotFoundError:

        iter_start = 0 
        batch_start = 0 
        pass 

    print(f"Trainign starting from iteration {iter_start}, batch {batch_start}")

    number_observations = input_tensor.shape[0]

    n_complete_batches = number_observations//batch_size 
    n_batches = n_complete_batches + 1

    print(f"There are {n_batches} batches per iteration")

    with trange(iter_start, n_epochs) as tr:
        iter_losses = []
        for it in tr:

            batch_losses = []
            
            plt.figure()
            plt.xlabel(f'The number of batches for iteration {it}')
            plt.ylabel('Training loss (Negative Log likelihood')

            print(f'Trainign for Iteration: {it} starts')
            
            for b in range(batch_start, n_batches):

                print(f'Training for Iteration: {it} Batch {b} starts')

                #print(f'Iteration: {it} - Batch: {b}')
                no_child = input_tensor.shape[-3]

                ### select the dataset according to the batch size
                if b <= n_complete_batches-1:
                    input_batch = input_tensor[b*batch_size : (b+1) * batch_size]
                    target_batch = target_tensor[b*batch_size : (b+1) * batch_size]
                else: 
                    input_batch = input_tensor[b*batch_size:]
                    target_batch = target_tensor[b*batch_size:]

                decoder_batch_ouputs = zeros(input_batch.shape[0], target_len, no_child, model.lstm_output_dim,)
                attention_batch_outputs = zeros(input_batch.shape[0], target_len, no_child, model.mha_output_dim,)
                model_batch_outputs = zeros(input_batch.shape[0], target_len, no_child, model.model_ouput_dim,)

                batch_loss = 0.
                batch_loss_no_grad = 0.
                "---- ZERO GRAD ----"
                optimizer.zero_grad()
                # pass in each observation for forward propogation
                for batch_index in range(input_batch.shape[0]):
                
                    input = input_batch[batch_index, :, :, :]
                    target = target_batch[batch_index, :, :, :] 
                    #print(input.shape)

                    output, decoder_ouputs, value = model.forward(target_len, input)
                    decoder_batch_ouputs[batch_index] = decoder_ouputs 
                    attention_batch_outputs[batch_index] = value
                    model_batch_outputs[batch_index] = output 

                    output = torch.squeeze(output, dim=-1)
                    target = torch.squeeze(target, dim=-1)
                    # print(output.shape)
                    # print(target.shape)
                    target = torch.permute(target, (1,0))

                    #print(output)
                    #print(target)
                    loss = get_loss(output, target, 0.000001)
                    #print(torch.exp(-loss))

                    #print(output[0])
                    #print(target[0])
                    #mini_drichilet = Dirichlet(output[0])
                    #mini_pdf = mini_drichilet.log_prob(target[0],0.000001)
                    #batch_pdf.append(np.exp(mini_pdf.detach().numpy()))
                
                    # reduction method defulat to sum, instead of mean 
                    batch_loss += loss
                    batch_loss_no_grad += loss.item()

                "---- BackProp ----"
                average_batch_loss = batch_loss/batch_size
                average_batch_loss.backward() 
            
                if clip: 
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=2)
                optimizer.step()
                
                batch_losses.append(average_batch_loss)

                "---- Tracing  ----"
                if tracing:
                    if batch_loss_no_grad >= 20: 
                        batch_loss_no_grad = 20

                    plt.plot(
                        b,
                        (batch_loss_no_grad/batch_size),
                        'rs',
                    )

                if b%20 == 0 and b!= 0: 
                #   print(f"The loss for iteration {it} batch {b} is {batch_loss_no_grad/batch_size}")
                    "---- Saving ----- "
                    plt.show()
                    torch.save(
                        {   'lr' : learning_rate,
                            'epoch': it,
                            'batch': b,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': batch_losses,
                        }, 
                        PATH
                    )
                    "--- Validation ---- "
                    model.eval()
                    valid_ouput = model.forward(val_input_tensor)
                    # C, F 
                    distributions = Dirichlet(valid_ouput)
                    




            #print(f"Training for iteration {it} is completed")
            iter_loss = sum(batch_losses)/n_batches 
            #print(f"The average loss for iteration {it} is {iter_loss}")
            tr.set_postfix(loss="{0:.3f}".format(iter_loss))

        iter_losses.append(iter_loss)
    
    return model, iter_losses



def train_model_test(
    model, 
    input_tensor, 
    target_tensor, 
    n_epochs, 
    target_len, 
    batch_size, 
    learning_rate,
    PATH = "model.pt",
    clip = False,
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

    try: 
        checkpoint = torch.load(PATH) 

        if learning_rate == checkpoint['lr']: 

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 

            iter_start = checkpoint['epoch']
            batch_start = checkpoint['batch']

        else: 
            iter_start = 0 
            batch_start = 0 

    except FileNotFoundError:

        iter_start = 0 
        batch_start = 0 
        pass 

    print(f"Trainign starting from iteration {iter_start}, batch {batch_start}")

    number_observations = input_tensor.shape[0]

    n_complete_batches = number_observations//batch_size 
    n_batches = n_complete_batches + 1

    print(f"There are {n_batches} batches per iteration")

    with trange(iter_start, n_epochs) as tr:
        iter_losses = []
        for it in tr:

            batch_losses = []
            batches = []
            #plt.figure()

            print(f'Trainign for Iteration: {it} starts')
            
            for b in range(batch_start, n_batches):

                print(f'Training for Iteration: {it} Batch {b} starts')

                #print(f'Iteration: {it} - Batch: {b}')
                no_child = input_tensor.shape[-3]

                ### select the dataset according to the batch size
                if b <= n_complete_batches-1:
                    input_batch = input_tensor[b*batch_size : (b+1) * batch_size]
                    print(input_batch.shape)
                    target_batch = target_tensor[b*batch_size : (b+1) * batch_size]
                else: 
                    input_batch = input_tensor[b*batch_size:]
                    target_batch = target_tensor[b*batch_size:]
                    print(input_batch.shape)

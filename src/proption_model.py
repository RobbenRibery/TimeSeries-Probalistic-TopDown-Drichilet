from ast import For
from random import paretovariate
from tkinter import N
from matplotlib.pyplot import axis
from torch.nn import (
    LSTM, 
    MultiheadAttention, 
    Linear
)
from torch import tensor, rand, zeros, matmul, permute, log, lgamma, cat
from torch import nn
from torch import optim 
import torch.nn.functional as F
from dirichlet import * 
import numpy as np 
import math 

def get_loss(output, target): 

    drichilet = Dirichlet(output) 

    # print(f"the batch shape is {output.shape[:-1]}")
    # print(f"the event shape is {output.shape[-1:]}")
    # print(drichilet)

    loss = drichilet.log_prob(target)
    loss_sum = loss.sum(-1)

    return -loss_sum

class hts_embedding(nn.Module): 

    """
    The embedding class for the time series 
    input: 
    - num_embeeddings and the dimension of embeddings 
    - x: the input that represents the index of the time series, starting from zero 

    """

    def __init__(self, num_embedd, embedd_dim, ) -> None:
        super(hts_embedding,self).__init__()

        self.num_embedd = num_embedd 
        self.embedd_dim = embedd_dim 
        self.embedd_layer = nn.Embedding(self.num_embedd, self.embedd_dim, max_norm=True)

    def forward(self, x): 

        output = self.embedd_layer(x)

        return output    

# unidirectional implementation of the LSTM Seq-2-Seq Network 
# Author - Rundong Liu 
class encoder_lstm(nn.Module): 

    ''' Encodes time-series sequence '''

    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_num_layer, batch_first=False) -> None:

        super(encoder_lstm, self).__init__()

        '''
        : param lstm_input_dim:     the number of features in the input X
        : param lstm_hidden_dim:    the number of features in the hidden state h
        : param lstm_num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        : batch_first:          True --> first input would be the number of sequences/observations
        '''

        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layer = lstm_num_layer 
        self.batch_first = batch_first 

        self.lstm = LSTM(self.lstm_input_dim, self.lstm_hidden_dim, self.lstm_num_layer, batch_first = self.batch_first ,)

    def init_hidden(self, batch_size):
        
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        return (zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim),
                zeros(self.lstm_num_layer, batch_size, self.lstm_hidden_dim))

    def forward(self, x, h0, c0): 

        '''
        : param x :                    input of shape (# in batch, seq_len, lstm_input_dim) 

        : return lstm_out,:            lstm_out gives all the hidden states in the sequence;
        
        : return self.cache            gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''

        encoder_ouptut, (self.hn, self.cn) = self.lstm(x, (h0, c0))
        self.cache = (self.hn, self.cn)

        return encoder_ouptut, self.cache 

    
class decoder_lstm(nn.Module): 

    " Decodes time-series sequence"

    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_num_layer, ouptut_dim, batch_first = False) -> None:
        super(decoder_lstm, self).__init__()

        '''
        : param self.lstm_input_dim:     the number of features in the input X
        : param lstm_hidden_dim:    the number of features in the hidden state h
        : param lstm_num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        : batch_first:          True --> first input would be the number of sequences/observations
        '''

        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim 
        self.lstm_num_layer = lstm_num_layer 
        self.lstm_output_dim = ouptut_dim
        self.batch_first = batch_first 
        self.lstm = LSTM(self.lstm_input_dim, self.lstm_hidden_dim, self.lstm_num_layer, batch_first= self.batch_first,)
        self.linear = Linear(self.lstm_hidden_dim, self.lstm_input_dim) 
        self.linear_final = Linear(self.lstm_hidden_dim, self.lstm_output_dim)

    def forward(self, x, encoder_cache): 

        """

        x: input at the last timestamp  input of shape should 2D (batch_size, input_size)
    
        encoder_output: cache of the encoder output 
            hn: (D time lstm_num_layers, # nathces, lstm_hidden_dim))
            cn: (D time lstm_num_layers, # nathces, lstm_hidden_dim))
            D = 2 if bi-directional, 1 if unidirectional 
        """
        (hn, cn) = encoder_cache

        if self.batch_first: 
            x = x.unsqueeze(1) 

        else: 
            x = x.unsqueeze(0) 

        decoder_lstm_output, (self.hn , self.cn) = self.lstm(x, (hn, cn))
        self.cache = (self.hn , self.cn)

        if self.batch_first: 
            decoder_lstm_output = decoder_lstm_output.squeeze(1) 
        else: 
            decoder_lstm_output = decoder_lstm_output.squeeze(0) 

        decoder_ouput = self.linear(decoder_lstm_output) 

        return decoder_ouput, self.cache


class mha_novel(nn.Module): 

    """
    The Multi Head Attention Layer sitting on top of the decoding network 
    
    """

    def __init__(self, lstm_input_dim, hts_embedd_dim, num_head) -> None:
        super().__init__()

        """
        hts_embedd_dim: the number of feature i
        lstm_input_dim: the number of features in the input x  

        """
        assert hts_embedd_dim % num_head == 0 
        # hts_embedd_dim / num_head = query size 
        # each head learns part of the embedding dimension to make sure 
        # that specific questions are asked with each head 

        self.hts_embedd_dim = hts_embedd_dim
        self.num_head = num_head 
        self.query_size = hts_embedd_dim // num_head

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = Linear(lstm_input_dim, 3*hts_embedd_dim)
        self.o_proj = Linear(hts_embedd_dim, hts_embedd_dim)

        self._reset_parameters()

    def _reset_parameters(self):

        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False): 

        """
        x: input tensor with (batch size, length of the sequence, hts_embedd_dim)
        """

        # calcualte WQ, WK and WV together and seperate them into three seperate matrics 
        batch_size, seq_length, hts_embedd_dim = x.size() 
        qkv = self.qkv_proj(x) 
        qkv = qkv.reshape(batch_size, self.num_head, seq_length, 3*self.query_size)
        q, k, v = qkv.chunk(3, dim=-1) 
        """
        q: batch size, num_head, seq_length, query_size 
        k: batch size, num_head, seq_length, query_size 
        v: batch size, num_head, seq_length, query_size   
        """

        q_dot_k = matmul(q, k.transpose(-2, -1)) 
        scaled_q_dot_k = q_dot_k/math.sqrt(self.query_size) 

        """
        scaled_q_dot_k: batch size, num_head, seq_length, seq_length
        """

        if mask is not None:
            scaled_q_dot_k  = scaled_q_dot_k.masked_fill(mask == 0, -9e15)

        attention = nn.functional.softmax(scaled_q_dot_k, dim = -1)

        """
        attention: batch size, num_head, seq_length, seq_length
        """

        value = matmul(attention,v)
        #print(value.shape)
        """
        value: batch size, num_head, seq_length, query_size
        """
        
        value = value.permute(0, 2, 1, 3)
        #print(value.shape)
        """
        value: batch size, seq_length, num_head, query_size
        """

        value = value.reshape(batch_size, seq_length, hts_embedd_dim)
        """
        value: batch size, seq_length, hts_embedd_dim
        """

        value = self.o_proj(value)

        # note: here we reutrn a value matrix that has dimension 
        # the same as the input dimension, so we can stack/loop the MHA abitraty number of times 

        if return_attention: 

            return value, attention

        else: 

            return value 

class mha(nn.Module): 

    def __init__(self, lstm_input_dim, hts_embedd_dim, num_head, batch_first = False) -> None:
        super().__init__()

        self.lstm_input_dim = lstm_input_dim 
        self.hts_embedd_dim = hts_embedd_dim 
        self.num_head = num_head 
        self.batch_first = batch_first 
        self.mha = MultiheadAttention(self.hts_embedd_dim, self.num_head, batch_first=self.batch_first)

    def forward(self, x): 

        values, atten_wieghts = self.mha(x, x, x)

        return values, atten_wieghts

class fc_skip(nn.Module): 

    def __init__(self, lstm_input_dim, lstm_output_dim, activation=nn.ReLU()) -> None:
        super().__init__()

        self.lstm_input_dim = lstm_input_dim 
        self.lstm_output_dim = lstm_output_dim 
        self.activation = activation
        self.fc = Linear(lstm_input_dim, lstm_output_dim)

    def forward(self, x):  

        output = self.fc(x)
        output += x 
        output = self.activation(output) 

        return output

class linear(nn.Module): 

    def __init__(self, lstm_input_dim, lstm_output_dim) -> None:
        super().__init__()

        self.lstm_input_dim = lstm_input_dim
        self.lstm_output_dim = lstm_output_dim 
        self.linear = nn.Linear(self.lstm_input_dim, self.lstm_output_dim)

    def forward(self, x): 

        output = self.linear(x)

        return output 


class proportion_model(nn.Module): 

    def __init__(
        self, 
        num_hts_embedd, hts_embedd_dim, 
        lstm_input_dim, lstm_hidden_dim, lstm_num_layer, lstm_output_dim, 
        mha_embedd_dim, num_attention_head, num_attention_layer, 
        num_embedd, embedd_dim
    ) -> None:
        super().__init__()

        """
        lstm_input_dim: input dimensiotn of the LSTM encoder 
        lstm_hidden_dim: fixed hidden dimension of the LSTM encoder and decoder 
        lstm_num_layer: number of the layers in the 
        
        
        """
        self.num_hts_embedd = num_hts_embedd
        self.hts_embedd_dim = hts_embedd_dim

        self.lstm_input_dim = lstm_input_dim 
        self.lstm_hidden_dim = lstm_hidden_dim 
        self.lstm_num_layer = lstm_num_layer 
        self.lstm_output_dim = lstm_output_dim

        self.mha_embedd_dim =  mha_embedd_dim 
        self.num_attention_head = num_attention_head
        self.num_attention_layer = num_attention_layer 
       
        self.embedd_layer = hts_embedding(self.num_hts_embedd, self.hts_embedd_dim)
        self.encoder_lstm = encoder_lstm(self.lstm_input_dim, self.lstm_hidden_dim, self.lstm_num_layer, batch_first=False)
        self.decoder_lstm = decoder_lstm(self.lstm_input_dim, self.lstm_hidden_dim, self.lstm_num_layer, self.lstm_output_dim , batch_first=False)

        self.mha = mha(self.lstm_output_dim, self.mha_embedd_dim, self.num_attention_head, batch_first=False)
        self.fc_skip = fc_skip(self.lstm_output_dim,self.lstm_output_dim)
        self.linear = Linear(self.lstm_output_dim,1)

    def train(
        self, 
        input_tensor, target_tensor, 
        n_epochs, 
        target_len, batch_size, 
        learning_rate = 0.01,
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

        losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr = learning_rate)

        n_batches = int(input_tensor.shape[0] / batch_size)

        for i in range(n_epochs):

            batch_loss = 0  

            for b in range(n_batches):

                print(f"Training Batch number: {b}")

                ### select the dataset according to the batch size 
                input_batch_whole = input_tensor[b : b + batch_size, :, :]   # history proportions for each children 

                input_batch = input_batch_whole[b : b + batch_size, :, -1]
                hts_embedding_input = input_batch_whole[b : b + batch_size, :, :-1].squeeze()
                print(f"The input batch diemsion is {input_batch.shape}")
                print(f"The batch input is {input_batch}")
                target_batch = target_tensor[b : batch_size, :, :] # future proportions for each children
                print(f"The input target diemsion is {target_batch.shape}")

                ### get the time series embedding encoding 
                embedd_vector = hts_embedding(hts_embedding_input)

                ### append the embed vector into the model 
                input_batch = cat((input_batch, embedd_vector) , 0) 

                ### pass through the encoder & decoder LSTM network
                ## encoder
                decoder_ouputs = zeros(batch_size, target_len, self.lstm_output_dim)
                # intitialise the LSTM encoder initial cell state and hidden state 
                h0, c0 = self.encoder_lstm.init_hidden(batch_size) 
                # zero the gradient
                optimizer.zero_grad()
                encoder_ouput, (encoder_hn, encoder_cn) = self.encoder_lstm.forward(input_batch, h0, c0)
                encoder_cache = (encoder_hn, encoder_cn)
                print(f"The encoder final hidden state shape is {encoder_hn.shape}")
                print(f"The encoder final cell state shape is {encoder_hn.shape}")

                ## Decoder
                decoder_input = input_batch[:,-1,:]
            
                print(f"The decoder input shape is {decoder_input.shape}")
                # forwrad prop through the time 
                for t in range(target_len): 

                    decoder_output, (decoder_hn, decoder_cn) = self.decoder_lstm.forward(decoder_input, encoder_cache)
                    decoder_ouputs[:,t,:] = decoder_output 

                    encoder_cache = (decoder_hn, decoder_cn)  
                    decoder_input = decoder_output 

                print(f'The decoder output dimension is {decoder_ouputs.shape}')
                decoder_ouputs = decoder_ouputs.view(
                    decoder_ouputs.shape[1],
                    decoder_ouputs.shape[0],
                    decoder_ouputs.shape[2]
                ) # batch dimension is sequence length 
                print(f'Putting sequence length as the batch numer, the decoder output dimension is {decoder_ouputs.shape}')

                # forward pass through n layers of attention 
                attention_inputs = decoder_ouputs
                print(f'Attention input has shape : {attention_inputs.shape}')
                for i in range(self.num_attention_layer):
                    
                    value, attention_weights  = self.mha.forward(attention_inputs)
                    value = self.fc_skip.forward(value)
                    attention_inputs = value 
                    #print(attention_inputs.shape)

                print(f"The value-ouput from the attention layer has shape {value.shape} ")
    
                batch_output = self.linear(value) # L, C, 1

                print(f"The value-ouput from the linear layer has shape {batch_output.shape} ")

                print(batch_output)
               
                print(f"The ouput from the model has shape {batch_output.shape}")
                #print(f"The ouput from the model is {batch_output}")

                loss = get_loss(batch_output, target_batch)

                batch_loss += loss

            batch_loss.backward()

            optimizer.step()
    
        return 



if __name__ == "__main__": 

    """
    Data Processing Script 
    
    """

    # dimension about the dataset
    no_child = 8
    History = 24 
    Forward = 12 
    covariate_dim = 4 

    lstm_input_dim = 3
    embediing_dim = 12
    lstm_lstm_hidden_dim = 48
    lstm_layer_size = 1
    number_attention_head = 4
    number_attention_layer = 1
    lstm_input_dim = 3*History+embediing_dim
    lstm_output_dim = lstm_input_dim
    batch_size = 2


    time_dimension = (History+Forward) * 3
    # proportions matrix T * C 
    proportions = F.softmax(torch.rand(time_dimension, no_child),)
    print(proportions.shape)

    # parent sales matrix T*1
    parent  = torch.randint(10,(time_dimension,1))
    print(parent.shape)
    
    # covariates matrix T * covariate_dim 
    covariate = torch.rand(time_dimension,covariate_dim)
    print(covariate.shape) 

    # hie_index
    hie_index = torch.arange(no_child)
    print(hie_index)

    proportions_3d = proportions.reshape(time_dimension, no_child, 1)
    print(proportions_3d.shape)
    print(proportions_3d[0,:,:].sum())

    parent_3d = (parent.unsqueeze(1)).expand(parent.shape[0], no_child, parent.shape[-1])
    print(parent_3d.shape)
    
    data_3d = torch.cat((proportions_3d, parent_3d), dim=-1)
    print(data_3d.shape)
    print(data_3d[0,:,:])

    covariate_3d = covariate.unsqueeze(1).expand(covariate.shape[0], no_child, covariate.shape[-1])
    print(covariate_3d.shape)
    print(covariate_3d[0,:,:])

    data_3d = torch.cat((data_3d,covariate_3d), dim=-1)
    print(data_3d.shape)
    print(data_3d[0,:,:])

    hie_index_2d = hie_index.expand(time_dimension, no_child)
    hie_index_3d = hie_index_2d.reshape(hie_index_2d.shape[0], hie_index_2d.shape[-1], 1)
    print(hie_index_3d.shape)
    print(hie_index_3d[0,:,:])

    data_3d = torch.cat((data_3d,hie_index_3d), dim=-1)
    print(data_3d.shape)
    print(data_3d[0,:,:])

    number_observations = data_3d.shape[0] - (History+Forward)+1 

    data_3d_time_batched = torch.empty(
        number_observations, 
        History+Forward,
        data_3d.shape[1],
        data_3d.shape[2]
    )

    for i in range(number_observations): 

        data_3d_time_batched[i,:,:,:] = data_3d[i:i+History+Forward, :, :] 


    print(data_3d_time_batched.shape)
    print(data_3d_time_batched[-1,:,:,:].shape)

    

    if torch.equal(data_3d_time_batched[-1,-1,:,:], data_3d[-1,:,:]): 


        print('data correctly processed to generate time-bacted tensor')

        input_tensor = torch.empty(
            number_observations, 
            History,
            data_3d_time_batched.shape[-2],
            data_3d_time_batched.shape[-1]
        )
        
        ## We first use the recursive predicitng mechanism in LSTM, in the future we release more blocks that adapt to teacher-forcing/mixed training 
        target_tensor = torch.empty(
            number_observations, 
            Forward,
            data_3d_time_batched.shape[-2],
            1
            # data_3d_time_batched.shape[-1]
        )

        for i in range(data_3d_time_batched.shape[0]): 

            input_tensor[i] = data_3d_time_batched[i,:History,:,:]
            target_2d = data_3d_time_batched[i,History:,:,0]
            target_tensor[i] =  target_2d.reshape(target_2d.shape[0], target_2d.shape[1],1)

            # print(input_tensor.shape)
            # print(target_tensor.shape)

        print(input_tensor.shape)
        print(target_tensor.shape)
        # print(target_tensor[-1,0,:,:].sum())

    else: 

        raise "Missed processed data point, not all time points are batched"



    


    # data on aggregate 
    # data = torch.cat((proportions,parent,covariate),axis=1)
    # print(data.shape)

    # #reshpe the data into 3-D first 
    # data_3d = data.view(time_dimension, no_child, data.shape[-1] )
    # print(data_3d.shape)

    # proportion_model = proportion_model(
    #     lstm_input_dim, lstm_lstm_hidden_dim, lstm_layer_size, lstm_output_dim, lstm_output_dim, number_attention_head, number_attention_layer,
    # )

    # input_tensor = rand(History, no_child,lstm_input_dim)
    # target_tensor = rand(no_child, Forward, 1)
    
    # proportion_model.train( 
    #     input_tensor, 
    #     target_tensor, 
    #     1, 
    #     Forward, 
    #     batch_size, 
    #     learning_rate = 0.01,)

    


        


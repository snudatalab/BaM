'''
***********************************************************************
Towards True Multi-interest Recommendation: Enhanced Scheme for Balanced Interest Training

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: BasicModel.py
- A basic model framework for all models.
- The function 'select_interest()' implements the proposed idea 'Soft-selection' of BaM.

Version: 1.0
***********************************************************************
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_, zeros_
from SSLoss import SSLoss


'''
Basic model framework for all models

input:
    * item_num: number of items
    * hidden_size: size of hidden layers
    * batch_size: size of the batch
    * seq_len: length of the sequence
'''
class BasicModel(nn.Module):

    # initialization of model
    def __init__(self, item_num, interest_num, hidden_size, batch_size, seq_len=50, selection=False, T=1, linear_size=16):
        super(BasicModel, self).__init__()
        self.name = 'base'
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.item_num = item_num
        self.seq_len = seq_len
        self.embeddings = nn.Embedding(self.item_num+1, self.hidden_size, padding_idx=0)
        self.interest_num = interest_num
        self.selection = selection
        self.T = T
        if selection == 'l': # defines linear layers needed for BaM_l
            self.linear_size = linear_size
            self.l1 = nn.Linear(self.hidden_size, linear_size, bias=True)
            self.l2 = nn.Linear(self.hidden_size, linear_size, bias=True)

    # sets used device
    def set_device(self, device):
        self.device = device

    # sets sampler for negative sample
    def set_sampler(self, sampled_n, beta=0, device=None):
        
        self.is_sampler = True
        if sampled_n == 0:
            self.is_sampler = False
            return
        
        self.sampled_n = sampled_n
        
        noise = self.build_noise(self.item_num+1)
        self.sample_loss = SSLoss(noise=noise,
                                       noise_ratio=self.sampled_n,
                                       beta=beta,
                                       device=device
                                       )

    # initialization of weights
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        elif isinstance(module, nn.MultiheadAttention):
            xavier_normal_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                zeros_(module.in_proj_bias)
            xavier_normal_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                zeros_(module.out_proj.bias)


    # parameters reset
    def reset_parameters(self, initializer=None):
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.kaiming_normal_(weight)
            else:
                zeros_(weight)

    '''
    PROPOSED: Soft-selection
    
    input:
        * interest_eb: multi-interest representation
        * pos_eb: embedding of positive item
    returns:
        * selection: selected interest
    '''
    def select_interest(self, interest_eb, pos_eb): 
            
        # BaM_p: calculates the correlation score between interests and positive item by simple inner-product
        if self.selection == 'p':
            corr = torch.matmul(interest_eb, torch.reshape(pos_eb, (-1, self.hidden_size, 1))).squeeze() # inner-product of multi-interest representation and 
            prob = F.softmax(self.T*corr, dim=1)
            selected_index = torch.multinomial(prob, 1).flatten() # selecting an interest based on the probability
            selection = torch.reshape(interest_eb, (-1, self.hidden_size))[
                    (selected_index + torch.arange(pos_eb.shape[0], device=interest_eb.device) * self.interest_num).long()]
        
        # Bam_l: calculates the correlation score between interests and positive item by passing each of them to a linear layer
        elif self.selection == 'l':
            comp_interest = self.l1(interest_eb) # gets important features from multi-interest representation
            comp_pos = self.l2(pos_eb.reshape(-1, 1, self.hidden_size)) # gets importatnt features from postivie item embedding
            corr = torch.matmul(comp_interest, torch.reshape(comp_pos, (-1, self.linear_size, 1))).squeeze() # inner-product of compressed multi-interest representation and embedding of positive item (Equation (9))       
            prob = F.softmax(corr*self.T, -1).squeeze()
            selected_index = torch.multinomial(prob, 1).flatten()
            selection = torch.reshape(interest_eb, (-1, self.hidden_size))[
                        (selected_index + torch.arange(pos_eb.shape[0], device=interest_eb.device) * self.interest_num).long()]
          
        # hard selection from previous methods 
        else: 
            corr = torch.matmul(interest_eb, torch.reshape(pos_eb, (-1, self.hidden_size, 1))) # inner-product of multi-interest representation and embedding of positive item (Equation (8))
            prob = F.softmax(torch.reshape(corr, (-1, self.interest_num)), dim=-1) # probability of interest selection (Equation (7))
            selected_index = torch.argmax(prob, dim=-1) # hard interest selection using argmax
            selection = torch.reshape(interest_eb, (-1, self.hidden_size))[
                        (selected_index + torch.arange(pos_eb.shape[0], device=interest_eb.device) * self.interest_num).long()]
            
        return selection

    # calculates recommendation score for all items given multi-interest representation
    def calculate_score(self, interest_eb):
        all_items = self.embeddings.weight
        scores = torch.matmul(interest_eb, all_items.transpose(1, 0)) # [b, n]
        return scores

    # calculates sampled softmax loss
    def calculate_sampled_loss(self, selection, pos_items, multi_interest=False):
        return self.sample_loss(pos_items.unsqueeze(-1), selection, self.embeddings.weight, multi_interest)

    # generates noise for negative sampling
    def build_noise(self, number):
        total = number
        freq = torch.Tensor([1.0] * number).to(self.device)
        noise = freq / total 
        assert abs(noise.sum() - 1) < 0.001
        return noise
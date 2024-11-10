'''
***********************************************************************
Towards True Multi-interest Recommendation: Enhanced Scheme for Balanced Interest Training

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: SSLoss.py
- A class of sampled softmax loss.
- This code is based on the implementation of https://github.com/Stonesjtu/Pytorch-NCE/.
- The function 'get_multi_score()' and 'forward()' implements the proposed idea 'Multi-interest Loss' of BaM.

Version: 1.0
***********************************************************************
'''

import torch
import torch.nn as nn
BACKOFF_PROB = 1e-10
    
    
"""
Sampled softmax approximation

inputs:
    * noise: the distribution of noise
    * noise_ratio: $\frac{#noises}{#real data samples}$
    * target: the supervised training label
returns:
    * the scalar loss ready for backward
"""
class SSLoss(nn.Module):
    
    # initialization
    def __init__(self,
                 noise,
                 noise_ratio=100,
                 beta = 0,
                 device=None
                 ):
        super(SSLoss, self).__init__()
        self.device = device
        # Re-norm the given noise frequency list and compensate words with
        # extremely low prob for numeric stability
        self.update_noise(noise)
        self.noise = noise

        self.noise_ratio = noise_ratio
        self.beta = beta
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')

    # udate the noise
    def update_noise(self, noise):
        probs = noise / noise.sum()
        probs = probs.clamp(min=BACKOFF_PROB)
        renormed_probs = probs / probs.sum()
        self.register_buffer('logprob_noise', renormed_probs.log())

    # compute the loss with output and the desired target
    def forward(self, target, selection, embs, multi_interest=False):
        batch = target.size(0)

        noise_samples = torch.arange(embs.size(0)).to(self.device).unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1) if self.noise_ratio == 1 else self.get_noise(batch)

        logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
        logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

        logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
        logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

        logit_target_in_model, logit_noise_in_model = self.get_score(target, noise_samples, selection, embs)

        loss = self.sampled_softmax_loss(
                logit_target_in_model, logit_noise_in_model,
                logit_noise_in_noise, logit_target_in_noise,
            )
        
        # If using the multi-interest loss:
        if isinstance(multi_interest, torch.Tensor):
            logit_target_in_model, logit_noise_in_model = self.get_multi_score(target, noise_samples, embs, multi_interest)
            loss += self.sampled_softmax_loss(
                logit_target_in_model, logit_noise_in_model,
                logit_noise_in_noise, logit_target_in_noise,
            ) # calculates multi-interest loss and add it with single-interest loss (see Equation (12) and (13))

        return loss.mean()

    # Generate noise samples from noise distribution
    def get_noise(self, batch_size):
        noise_samples = torch.multinomial(self.noise.repeat((batch_size,1)), self.noise_ratio).unsqueeze(1)
        return noise_samples

    # Get the target and noise score.
    def get_score(self, target_idx, noise_idx, selection, embs):
        original_size = target_idx.size()
        selection = selection.contiguous().view(-1, selection.size(-1))
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[:, 0]
        target_batch = embs[target_idx]
        noise_batch = embs[noise_idx]
        
        target_score = torch.sum(selection * target_batch, dim=1)
        noise_score = torch.bmm(noise_batch, selection.unsqueeze(2))
            
        return target_score.view(original_size), noise_score.view(*original_size, -1)
    
    '''
    PROPOSED: score for multi-interest Loss
    
    input:
        * target_idx: index of positive items
        * noise_idx: index of negative samples
        * embs: item embeddings of the model
        * interests: multi-interest representation
    returns:
        * target_score: score of positive items considering all interests
        * target_score: score of negative samples considering all interests
    '''
    def get_multi_score(self, target_idx, noise_idx, embs, interests):
        original_size = target_idx.size()
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[:, 0]
        
        all_score = torch.matmul(interests, embs.unsqueeze(0).transpose(1,2))
        all_score = torch.logsumexp(all_score, dim=1) # combines items' score from all interests with LogSumExp function (see Equation (10) and (11))
        target_score = all_score[torch.arange(original_size[0]), target_idx]
        noise_score = torch.gather(all_score, 1, noise_idx.to(self.device))
            
        return target_score.view(original_size), noise_score.view(*original_size, -1)

    # Compute the sampled softmax loss based on the tensorflow's impl
    def sampled_softmax_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        ori_logits = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        q_logits = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)
        logits = ori_logits - q_logits
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()

        if self.beta == 0:
            loss = self.ce(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            ).view_as(labels)
        else:
            x = ori_logits.view(-1, ori_logits.size(-1))
            x = x - torch.max(x, dim = -1)[0].unsqueeze(-1)
            pos = torch.exp(x[:,0])
            neg = torch.exp(x[:,1:])
            imp = (self.beta * x[:,1:] -  torch.max(self.beta * x[:,1:],dim = -1)[0].unsqueeze(-1)).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            if torch.isinf(reweight_neg).any() or torch.isnan(reweight_neg).any():
                import pdb; pdb.set_trace()
            Ng = reweight_neg

            stable_logsoftmax = -(x[:,0] - torch.log(pos + Ng))
            loss = torch.unsqueeze(stable_logsoftmax, 1)
        return loss
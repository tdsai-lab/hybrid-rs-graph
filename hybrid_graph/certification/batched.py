"""Exact row-incident BWGNN evaluation for batches of realized graph/feature draws.

Uses (A+I)H plus explicit incident-row changes before row normalization.
The two polynomial powers are shared across Bernstein filters, as in the model.
All Gaussian coordinates are drawn; target scope retains only the target row.
This couples the two legal channels without changing either marginal law.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


class BatchedBWGNN:
    def __init__(self, smoother):
        self.smoother=smoother
        self.model=smoother.base_classifier
        assert not self.model.training and self.model.order==2
        assert smoother.adj_norm_type=='row'
        assert smoother.feature_noise_scope in ('target','all')
        graph=smoother._adj_csr_clean.copy()
        assert np.all(graph.data==1)
        self.graph=graph
        plus=(graph+sp.eye(graph.shape[0],dtype=np.float32,format='csr')).tocsr()
        self.base=torch.sparse_csr_tensor(torch.tensor(plus.indptr,dtype=torch.int64,device='cuda'),
                  torch.tensor(plus.indices,dtype=torch.int64,device='cuda'),
                  torch.tensor(plus.data,dtype=torch.float32,device='cuda'),size=plus.shape)
        self.degree=torch.tensor(np.asarray(plus.sum(axis=1)).ravel(),device='cuda',dtype=torch.float32)
        self.features=smoother.features

    def draw(self,node,count):
        n=self.features.shape[0]
        original=torch.tensor(self.graph.getrow(node).toarray().ravel(),device='cuda')
        keep=torch.rand((count,n),device='cuda')<self.smoother.prob
        bits=torch.randint(0,2,(count,n),device='cuda',dtype=torch.int32).float()
        row=torch.where(keep,original[None,:],bits)
        row[:,node]=0
        noise=torch.randn((n,count,self.features.shape[1]),device='cuda')*self.smoother.sigma
        if self.smoother.feature_noise_scope=='target':
            noisy=self.features[:,None,:].expand(-1,count,-1).clone()
            noisy[node]+=noise[node]
        else:
            noisy=self.features[:,None,:]+noise
        return row,noisy

    @torch.no_grad()
    def logits(self,node,rows,features):
        count=rows.shape[0];n=features.shape[0]
        original=torch.tensor(self.graph.getrow(node).toarray().ravel(),device='cuda')
        delta=(rows-original[None,:]).T.contiguous()
        degree=self.degree[:,None]+delta
        degree[node]=self.degree[node]+delta.sum(dim=0)
        assert torch.all(degree>=1)
        h=F.relu(self.model.linear(features))
        h=F.relu(self.model.linear2(h))
        width=h.shape[-1]
        def propagate(value):
            sums=torch.sparse.mm(self.base,value.reshape(n,count*width)).reshape(n,count,width)
            sums[node]+=torch.einsum('nb,nbh->bh',delta,value)
            sums+=delta[:,:,None]*value[node][None,:,:]
            return sums/degree[:,:,None]
        powers=[h[node]]
        value=h
        for _ in range(self.model.order):
            value=value-propagate(value)
            powers.append(value[node])
        filters=[sum(theta[k]*powers[k] for k in range(3)) for theta in self.model.theta]
        result=self.model.linear4(F.relu(self.model.linear3(torch.cat(filters,dim=-1))))
        return F.log_softmax(result,dim=-1)

    def sample_votes(self,node_idx,n_samples,batch_size):
        counts=np.zeros(2,dtype=np.int64)
        for start in range(0,n_samples,batch_size):
            rows,features=self.draw(node_idx,min(batch_size,n_samples-start))
            labels=self.logits(node_idx,rows,features).argmax(-1)
            counts+=torch.bincount(labels,minlength=2).cpu().numpy()
        return counts


class BatchedGCN:
    def __init__(self, smoother):
        self.smoother = smoother; self.model = smoother.base_classifier
        assert not self.model.training and self.model.num_layers == 2
        assert smoother.adj_norm_type == 'row' and smoother.feature_noise_scope == 'all'
        self.graph = smoother._adj_csr_clean.copy()
        plus = (self.graph + sp.eye(self.graph.shape[0],dtype=np.float32,format='csr')).tocsr()
        self.base = torch.sparse_csr_tensor(
            torch.tensor(plus.indptr,dtype=torch.int64,device='cuda'),
            torch.tensor(plus.indices,dtype=torch.int64,device='cuda'),
            torch.tensor(plus.data,dtype=torch.float32,device='cuda'),size=plus.shape)
        self.degree = torch.tensor(np.asarray(plus.sum(axis=1)).ravel(),device='cuda')
        self.features = smoother.features

    def draw(self, node, count):
        n = self.features.shape[0]
        original = torch.tensor(self.graph.getrow(node).toarray().ravel(),device='cuda')
        keep = torch.rand((count,n),device='cuda') < self.smoother.prob
        bits = torch.randint(0,2,(count,n),device='cuda',dtype=torch.int32).float()
        rows = torch.where(keep,original[None,:],bits); rows[:,node] = 0
        features = self.features[:,None,:] + torch.randn(
            (n,count,self.features.shape[1]),device='cuda') * self.smoother.sigma
        return rows, features

    @torch.no_grad()
    def logits(self,node,rows,features):
        n,count = features.shape[:2]
        original = torch.tensor(self.graph.getrow(node).toarray().ravel(),device='cuda')
        delta = (rows-original[None,:]).T.contiguous()
        degree = self.degree[:,None] + delta
        degree[node] = self.degree[node] + delta.sum(dim=0)
        assert torch.all(degree >= 1)
        support = features @ self.model.gc1.weight
        width = support.shape[-1]
        hidden = torch.sparse.mm(self.base,support.reshape(n,count*width)).reshape(n,count,width)
        hidden[node] += torch.einsum('nb,nbh->bh',delta,support)
        hidden += delta[:,:,None] * support[node][None,:,:]
        hidden /= degree[:,:,None]
        if self.model.gc1.bias is not None: hidden += self.model.gc1.bias
        hidden = F.relu(hidden)
        weights = rows.clone(); weights[:,node] = 1
        root = torch.einsum('bn,nbh->bh',weights,hidden) / degree[node,:,None]
        result = root @ self.model.gc2.weight
        if self.model.gc2.bias is not None: result += self.model.gc2.bias
        return F.log_softmax(result,dim=-1)

    def sample_votes(self,node_idx,n_samples,batch_size):
        counts = np.zeros(self.smoother.num_classes,dtype=np.int64)
        with torch.no_grad():
            for offset in range(0,n_samples,batch_size):
                rows,features = self.draw(node_idx,min(batch_size,n_samples-offset))
                labels = self.logits(node_idx,rows,features).argmax(-1)
                counts += torch.bincount(labels,minlength=len(counts)).cpu().numpy()
        return counts


def enable_batched_sampling(smoother):
    """Use the exact CUDA batched evaluator for the paper's supported models."""
    if not smoother.features.is_cuda or smoother.adj_norm_type != 'row':
        return False
    from hybrid_graph.architecture import BWGNN, GCN
    model = smoother.base_classifier
    if isinstance(model, BWGNN) and model.order == 2:
        sampler = BatchedBWGNN(smoother)
    elif isinstance(model, GCN) and model.num_layers == 2 and smoother.feature_noise_scope == 'all':
        sampler = BatchedGCN(smoother)
    else:
        return False
    smoother.sample_votes_target = sampler.sample_votes
    return True

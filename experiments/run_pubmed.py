#!/usr/bin/env python3
"""Train PubMed's two-layer GCN and certify joint incident-edge/feature budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from hybrid_graph.architecture import GCN
from hybrid_graph.certification import SmoothHybrid
from hybrid_graph.certification.batched import enable_batched_sampling
from hybrid_graph.data import load_data_new, normalize, sparse_mx_to_torch_sparse_tensor
from experiments.run_mgtab import sample_nodes, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('outdir', type=Path)
    p.add_argument('--dataset', choices=['pubmed'], default='pubmed')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--num_layers', type=int, choices=[2], default=2)
    p.add_argument('--model_arch', choices=['gcn'], default='gcn')
    p.add_argument('--dropout', type=float, default=.5)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--weight_decay', type=float, default=.0005)
    p.add_argument('--feature_noise', choices=['gaussian'], default='gaussian')
    p.add_argument('--sigma_train', type=float, default=.04)
    p.add_argument('--feature_noise_scope', choices=['all','target'], default='all')
    p.add_argument('--prob', type=float, default=.7)
    p.add_argument('--sigma', type=float, default=.01)
    p.add_argument('--skip_train', action='store_true')
    p.add_argument('--skip_certify', action='store_true')
    p.add_argument('--checkpoint_in', type=Path)
    p.add_argument('--N0', type=int, default=10)
    p.add_argument('--N', type=int, default=10000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--alpha', type=float, default=.001)
    p.add_argument('--skip', type=int, default=1)
    p.add_argument('--max', type=int, default=1000)
    p.add_argument('--certify_node_selection', choices=['random_stratified'], default='random_stratified')
    p.add_argument('--certify_sample_seed', type=int, default=42)
    p.add_argument('--d_list', default='0,1,2,3')
    p.add_argument('--eps_max', type=float, default=.1)
    p.add_argument('--n_eps', type=int, default=200)
    p.add_argument('--cert_mode', choices=['tight'], default='tight')
    args = p.parse_args()
    if min(args.epochs,args.N0,args.N,args.batch,args.max,args.n_eps)<1:
        p.error('Sample, epoch, batch, target and grid counts must be positive.')
    device = torch.device('cuda' if args.cuda else 'cpu')
    if args.cuda and not torch.cuda.is_available():p.error('CUDA is unavailable.')
    np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.cuda:torch.cuda.manual_seed_all(args.seed)
    graph,x,y,train,val,test = load_data_new('pubmed')
    graph=graph.tocsr();graph.setdiag(0);graph.eliminate_zeros()
    adj=sparse_mx_to_torch_sparse_tensor(normalize(graph+sp.eye(graph.shape[0]))).coalesce().to(device)
    binary=sparse_mx_to_torch_sparse_tensor(graph).coalesce().to(device)
    x,y=x.to(device),y.to(device);train,val=train.to(device),val.to(device)
    model=GCN(x.shape[1],args.hidden,int(y.max())+1,args.dropout).to(device)
    args.outdir.mkdir(parents=True,exist_ok=True)
    checkpoint=args.checkpoint_in or args.outdir/f'checkpoint.hybrid.prob.{args.prob}.sigma.{args.sigma}.pt'
    if args.skip_train:
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(saved.get('state_dict',saved))
    else:
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            model.train();optimizer.zero_grad()
            noisy=x+torch.randn_like(x)*args.sigma_train
            loss=F.nll_loss(model(noisy,adj)[train],y[train]);loss.backward();optimizer.step()
            if epoch==0 or (epoch+1)%25==0:print('epoch',epoch+1,'loss',float(loss.detach()),flush=True)
        # The PubMed driver saves the final-epoch checkpoint.
        torch.save({'state_dict':model.state_dict(),'epoch':args.epochs},checkpoint)
    model.eval()
    if args.skip_certify:return
    smoother=SmoothHybrid(model,int(y.max())+1,args.prob,binary,x,args.sigma,args.cuda,
                          args.feature_noise_scope,'row',adj)
    enable_batched_sampling(smoother)
    nodes=sample_nodes(y.cpu().numpy(),test.cpu().numpy(),args.max,args.certify_sample_seed)
    if not nodes:raise ValueError('Empty test population.')
    budgets=[int(v) for v in args.d_list.split(',')];eps=np.linspace(0,args.eps_max,args.n_eps)
    hit=np.zeros((len(budgets),len(eps)));seq=np.zeros_like(hit);cert=np.zeros_like(hit);rows=[]
    for i,node in enumerate(nodes):
        np.random.seed((args.seed*1000003+node*104729)%(2**31-1))
        torch.manual_seed((args.seed*1000003+node*104729)%(2**31-1))
        result=smoother.certify_node_hybrid(node,args.N0,args.N,args.alpha,args.batch,
            budgets,eps,(1-args.prob)/2,(1-args.prob)/2)
        pred=int(result['pred_class'][0]);correct=pred==int(y[node])
        hit+=result['grid_tight']*correct;seq+=result['grid_tight_seq']*correct;cert+=result['grid_tight']
        rows.append(dict(node=node,label=int(y[node]),pred=pred,pA_lower=float(result['pA_lower'][0]),
                         pB_upper=float(result['pB_upper'][0]),N=args.N,N0=args.N0))
        print('certified',i+1,'/',len(nodes),flush=True)
    summary=dict(dataset='pubmed',feature_noise_scope=args.feature_noise_scope,q=args.prob,sigma=args.sigma,
        N=args.N,N0=args.N0,alpha=args.alpha,n_targets=len(nodes),cert_mode='tight',
        d_list=budgets,eps_values=eps.tolist(),cert_acc_mode=(hit/len(nodes)).tolist(),
        cert_acc_seq_mode=(seq/len(nodes)).tolist(),frac_cert_mode=(cert/len(nodes)).tolist())
    (args.outdir/'hybrid_curve_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv(args.outdir/'per_node_stats.csv',rows)
    write_csv(args.outdir/'hybrid_curve.csv',[dict(d=d,epsilon=float(e),hybrid=float(hit[j,k]/len(nodes)),
        sequential=float(seq[j,k]/len(nodes))) for j,d in enumerate(budgets) for k,e in enumerate(eps)])


if __name__=='__main__':main()

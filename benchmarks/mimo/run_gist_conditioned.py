from __future__ import annotations
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from cortex_m.mimo.m2m_sparse_route import M2MSparseRoute, N_IN, N_OUT, ROLE_DIM, K, TAU_MIN
from cortex_m.mimo.gist_conditioned import GistConditionedM2M
from cortex_m.mimo.gist_direct_route import DirectGistRouteM2M

torch.set_num_threads(2)
TR, TE, EPOCHS, BS, LR = 4000, 1200, 500, 128, 0.0012
TAU_START, TAU_DECAY = 2.0, 0.985
SEEDS = [101,202,303,404,505,606,707,808,909,1001]
RULES = [[(0,1),(3,7),(4,5),(1,6)],[(2,7),(0,6),(1,4),(3,5)]]


def make_data(n, seed):
    gen=torch.Generator().manual_seed(seed); X=[]; R=[]; Q=[]; Y=[]; T=[]
    for _ in range(n):
        s=torch.randn(N_IN,generator=gen)
        qid=int(torch.randint(0,2,(1,),generator=gen).item()) # independent of x by design
        p=torch.randperm(N_IN,generator=gen); x=s[p]; r=F.one_hot(p,ROLE_DIM).float()
        y=[]; target=torch.zeros(N_OUT,N_IN)
        for j,(a,b) in enumerate(RULES[qid]):
            y.append(s[a]+s[b]); pa=(r[:,a]==1).nonzero(as_tuple=True)[0].item(); pb=(r[:,b]==1).nonzero(as_tuple=True)[0].item()
            target[j,pa]=1; target[j,pb]=1
        X.append(x); R.append(r); Q.append(F.one_hot(torch.tensor(qid),2).float()); Y.append(torch.stack(y)); T.append(target)
    return tuple(map(torch.stack,(X,R,Q,Y,T)))

XTR,RTR,QTR,YTR,TTR=make_data(TR,35001)
XTE,RTE,QTE,YTE,TTE=make_data(TE,36001)


def metrics(model, gist=None):
    model.eval()
    with torch.no_grad():
        if gist is None: y,gates,*_=model(XTE,RTE,TAU_MIN)
        else: y,gates,*_=model(XTE,RTE,gist,TAU_MIN)
    idx=gates.topk(K,dim=2).indices; selected=torch.zeros_like(gates); selected.scatter_(2,idx,1.)
    exact=((selected==(TTE>0.5)).all(dim=2).all(dim=1).float().mean().item())
    return {"MSE":F.mse_loss(y,YTE).item(),"Exact graph":exact,
            "Exact output":((selected==(TTE>0.5)).all(dim=2).float().mean().item())}


def train(model, seed, with_gist, epochs=EPOCHS):
    torch.manual_seed(seed); model.__init__() # same seed controls initialization for each condition
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    i0=torch.where(QTR[:,0]==1)[0]; i1=torch.where(QTR[:,1]==1)[0]
    for epoch in range(epochs):
        tau=max(TAU_MIN,TAU_START*(TAU_DECAY**epoch))
        a=i0[torch.randperm(len(i0))[:BS]]; b=i1[torch.randperm(len(i1))[:BS]]; ix=torch.cat([a,b])
        model.train()
        if with_gist: y,*_=model(XTR[ix],RTR[ix],QTR[ix],tau)
        else: y,*_=model(XTR[ix],RTR[ix],tau)
        loss=F.mse_loss(y,YTR[ix])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model


if __name__ == "__main__":
    variant=os.environ.get("MIMO_GIST_VARIANT","context").casefold()
    if variant not in {"context","direct"}: raise ValueError("MIMO_GIST_VARIANT must be context or direct")
    model_factory=GistConditionedM2M if variant=="context" else DirectGistRouteM2M
    out=Path(os.environ.get("MIMO_GIST_RESULTS_DIR","results/mimo-gist-conditioned")); out.mkdir(parents=True,exist_ok=True)
    (out/"checkpoints").mkdir(exist_ok=True)
    rows=[]
    for seed in SEEDS:
        print(f"seed={seed} no-gist baseline",flush=True)
        base=train(M2MSparseRoute(),seed,False); bm=metrics(base)
        print(f"seed={seed} gist-conditioned MIMO",flush=True)
        conditioned=train(model_factory(),seed,True)
        true=metrics(conditioned,QTE); flipped=metrics(conditioned,QTE.flip(1)); zero=metrics(conditioned,torch.zeros_like(QTE))
        row={"Seed":seed,"No-gist MSE":bm["MSE"],"No-gist exact graph":bm["Exact graph"],
             "True-gist MSE":true["MSE"],"True-gist exact graph":true["Exact graph"],
             "Flipped-gist MSE":flipped["MSE"],"Flipped-gist exact graph":flipped["Exact graph"],
             "Zero-gist MSE":zero["MSE"],"Zero-gist exact graph":zero["Exact graph"]}
        rows.append(row); print(json.dumps(row),flush=True)
        torch.save({"architecture":"GistConditionedM2M","variant":variant,"seed":seed,"state_dict":conditioned.state_dict(),
                    "config":{"N_IN":8,"N_OUT":4,"HIDDEN":32,"K":2,"TR":TR,"TE":TE,"EPOCHS":EPOCHS},"metrics":row},out/"checkpoints"/f"seed-{seed}.pt")
    df=pd.DataFrame(rows); df.to_csv(out/"seed_metrics.csv",index=False)
    summary={"benchmark":"M2M N=8 two task modes selected independently of numeric input; task mode supplied by retrieved gist",
     "created_utc":datetime.now(timezone.utc).isoformat(),"device":"cpu","torch_version":torch.__version__,
     "config":{"TR":TR,"TE":TE,"EPOCHS":EPOCHS,"BS":BS,"LR":LR,"seeds":SEEDS,"train_seed":35001,"test_seed":36001},
     "means":{k:float(df[k].mean()) for k in df.columns if k!="Seed"},
     "csv_sha256":hashlib.sha256((out/"seed_metrics.csv").read_bytes()).hexdigest(),
     "data_hashes":{"train":hashlib.sha256(XTR.numpy().tobytes()+RTR.numpy().tobytes()+QTR.numpy().tobytes()+YTR.numpy().tobytes()+TTR.numpy().tobytes()).hexdigest(),
                    "test":hashlib.sha256(XTE.numpy().tobytes()+RTE.numpy().tobytes()+QTE.numpy().tobytes()+YTE.numpy().tobytes()+TTE.numpy().tobytes()).hexdigest()},
     "variant":variant,"claim_limit":"Synthetic relational task only; causal perturbations establish sensitivity to the stored mode for this task, not general reasoning."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


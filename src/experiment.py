"""Run real dataset experiments; fixed train/validation/test split, no test selection."""
import sys,json,zipfile,hashlib,time,platform,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
import joblib
from core import ROOT, canonical, Agent,redact

def metrics(y,p):
    h=p>=.5
    return { 'accuracy':accuracy_score(y,h),'precision':precision_score(y,h,zero_division=0),'recall':recall_score(y,h,zero_division=0),'f1':f1_score(y,h,zero_division=0),'auc':roc_auc_score(y,p),'ap':average_precision_score(y,p),'confusion':confusion_matrix(y,h).tolist() }

def main():
    for d in ['models','results','figures']: (ROOT/d).mkdir(exist_ok=True)
    arc=ROOT/'data/uci_sms.zip'
    with zipfile.ZipFile(arc) as z:
        raw=z.read('SMSSpamCollection'); (ROOT/'data/SMSSpamCollection').write_bytes(raw)
        (ROOT/'data/UCI_readme.txt').write_bytes(z.read('readme'))
    rows=[line.split('\t',1) for line in raw.decode('utf8').splitlines() if line.strip()]
    df=pd.DataFrame(rows,columns=['label','text']); df['y']=(df.label=='spam').astype(int)
    df['key']=df.text.map(canonical)
    conflicts=df.groupby('key').y.nunique(); bad=set(conflicts[conflicts>1].index)
    clean=df[~df.key.isin(bad)].drop_duplicates('key').copy().reset_index(drop=True)
    clean['id']=clean.key.map(lambda t:hashlib.sha256(t.encode()).hexdigest()[:16])
    tr,rest=train_test_split(np.arange(len(clean)),test_size=.4,stratify=clean.y,random_state=42)
    va,te=train_test_split(rest,test_size=.5,stratify=clean.y.iloc[rest],random_state=42)
    X=clean.text; y=clean.y.to_numpy()
    configs={
      'word_nb':make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True),MultinomialNB(alpha=.1)),
      'word_lr':make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True),LogisticRegression(C=8,class_weight='balanced',max_iter=1000,random_state=42)),
      'char_lr':make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=40000),LogisticRegression(C=8,class_weight='balanced',max_iter=1000,random_state=42))}
    val={}; fit={}; probs={}; timings={}
    for name,pipe in configs.items():
        start=time.perf_counter(); pipe.fit(X.iloc[tr],y[tr]); timings[name]=time.perf_counter()-start
        val[name]=metrics(y[va],pipe.predict_proba(X.iloc[va])[:,1]); fit[name]=pipe
    best=max(val,key=lambda k:val[k]['f1']); pipe=fit[best]; pv=pipe.predict_proba(X.iloc[va])[:,1]
    # Two-sided reject option: empirical validation class purity; never a safety guarantee.
    lows=[t for t in np.arange(.01,.50,.01) if sum(pv<=t)>=30 and np.mean(y[va][pv<=t]==0)>=.99]
    highs=[t for t in np.arange(.50,1,.01) if sum(pv>=t)>=30 and np.mean(y[va][pv>=t]==1)>=.97]
    low=float(max(lows)) if lows else 0.; high=float(min(highs)) if highs else 1.
    bundle={'pipeline':pipe,'name':best,'thresholds':[low,high]}; joblib.dump(bundle,ROOT/'models/model.joblib')
    test={}; yt=y[te]
    for name,model in fit.items():
        probs[name]=model.predict_proba(X.iloc[te])[:,1];test[name]=metrics(yt,probs[name])
    probs['majority']=np.zeros(len(te));test['majority']=metrics(yt,probs['majority'])
    keyword=np.array([int(any(w in t.lower() for w in ['free','prize','winner','claim','cash'])) for t in X.iloc[te]])
    probs['keyword']=keyword;test['keyword']=metrics(yt,keyword)
    agent=Agent();start=time.perf_counter(); runs=[agent.run(t) for t in X.iloc[te]];ms=(time.perf_counter()-start)*1000/len(te)
    actions=np.array([r['action'] for r in runs]); accepted=np.isin(actions,['likely_ham','likely_spam']); guesses=(actions=='likely_spam').astype(int)
    def selective(mask,pred):
        return {'coverage':float(mask.mean()),'review_rate':float(1-mask.mean()),'accepted_accuracy':float(np.mean(pred[mask]==yt[mask])) if mask.any() else None,'false_normal':int(sum(mask&(pred==0)&(yt==1))),'false_spam':int(sum(mask&(pred==1)&(yt==0))),'accepted_n':int(sum(mask))}
    p=probs[best];mask=(p<=low)|(p>=high)
    ablations={'model_only':selective(np.ones(len(te),dtype=bool),(p>=.5).astype(int)),'reject_only':selective(mask,(p>=high).astype(int)),'full_agent':selective(accepted,guesses)}
    rng=np.random.default_rng(42); boot=[]
    for _ in range(1000):
        ix=rng.integers(0,len(te),len(te));boot.append(f1_score(yt[ix],p[ix]>=.5,zero_division=0))
    robust={}
    transformations={'uppercase':lambda t:t.upper(),'space_noise':lambda t:'  '.join(t.split()),'digit_mask':lambda t:__import__('re').sub(r'\d','#',t)}
    for name,fn in transformations.items():robust[name]=metrics(yt,pipe.predict_proba([fn(t) for t in X.iloc[te]])[:,1])
    out=pd.DataFrame({'id':clean.id.iloc[te].to_numpy(),'y':yt,'p':p,'prediction':(p>=.5).astype(int),'action':actions,'redacted_text':[redact(t) for t in X.iloc[te]]})
    out.to_csv(ROOT/'results/test_predictions.csv',index=False)
    out[out.y!=out.prediction].to_csv(ROOT/'results/errors.csv',index=False)
    pd.DataFrame([{'id':clean.id[i],'split':s,'label':int(y[i])} for s,idx in [('train',tr),('validation',va),('test',te)] for i in idx]).to_csv(ROOT/'results/splits.csv',index=False)
    # Model coefficients are statistical associations, not causal evidence.
    m=pipe[1]; c=m.coef_[0] if hasattr(m,'coef_') else m.feature_log_prob_[1]-m.feature_log_prob_[0]
    names=pipe[0].get_feature_names_out(); top=np.argsort(c)
    pd.DataFrame({'feature':names[np.r_[top[:15],top[-15:]]],'weight':c[np.r_[top[:15],top[-15:]]]}).to_csv(ROOT/'results/features.csv',index=False)
    # Cosine overlap diagnostic on chosen representation; no tuning on result.
    from sklearn.metrics.pairwise import linear_kernel
    sim=linear_kernel(pipe[0].transform(X.iloc[te]),pipe[0].transform(X.iloc[tr])); nearest=sim.max(axis=1)
    report={'dataset':{'raw':len(df),'clean':len(clean),'conflicting_groups':len(bad),'removed':len(df)-len(clean),'raw_spam':int(df.y.sum()),'clean_spam':int(clean.y.sum()),'sha256_archive':hashlib.sha256(arc.read_bytes()).hexdigest(),'sha256_member':hashlib.sha256(raw).hexdigest(),'splits':{s:{'n':len(ix),'spam':int(y[ix].sum())} for s,ix in [('train',tr),('validation',va),('test',te)]},'test_train_cosine_gt_09':int(sum(nearest>.9))},'selected':best,'thresholds':[low,high],'validation':val,'test':test,'ablations':ablations,'robustness':robust,'f1_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'agent_ms_per_item':ms,'train_seconds':timings,'environment':{'python':platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','scipy','joblib','matplotlib']}}}
    (ROOT/'results/metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    demos=['Congratulations! You won a FREE prize. Claim your cash now at https://bit.ly/example','Hi, I will meet you at the library after class.','您的快递异常，请点击链接。','', 'zxqv jkzx qvzx']
    (ROOT/'results/demo_trace.json').write_text(json.dumps([{'input':t,'output':agent.run(t)} for t in demos],ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

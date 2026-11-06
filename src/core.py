"""Offline, read-only SMS triage workflow. Never fetches a URL or changes messages."""
import re, json, hashlib
from pathlib import Path
from urllib.parse import urlsplit
import joblib
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def canonical(text):
    text = re.sub(r'https?://\S+|www\.\S+', '<url>', text.lower())
    return re.sub(r'\s+', ' ', re.sub(r'\d+', '<num>', text)).strip()

def inspect_links(text):
    links = re.findall(r'(?:https?://|www\.)[^\s<>]+', text, flags=re.I)
    out=[]
    for link in links:
        host=(urlsplit(link if '://' in link else 'https://'+link).hostname or '').lower()
        flags=[]
        if host in {'bit.ly','tinyurl.com','t.co','is.gd'}: flags.append('短链接目的地不透明')
        if re.fullmatch(r'\d+(?:\.\d+){3}',host): flags.append('数字IP地址')
        if host.startswith('xn--') or '.xn--' in host: flags.append('国际化域名需核验')
        out.append({'host':host,'flags':flags})
    return out

def redact(text):
    text=re.sub(r'https?://\S+|www\.\S+', '[LINK]', text)
    text=re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[EMAIL]',text)
    return re.sub(r'\b\d[\d\s+()-]{5,}\d\b','[NUMBER]',text)

class Agent:
    def __init__(self, model_path=None):
        self.error=None
        try: self.bundle=joblib.load(model_path or ROOT/'models/model.joblib')
        except Exception as exc: self.error=type(exc).__name__
    def run(self,text,tool=inspect_links):
        trace=[]
        def step(state,detail): trace.append({'state':state,'detail':detail})
        base={'action':'review','label':'需人工复核','score':None,'trace':trace,'links':[],'evidence':[]}
        if not isinstance(text,str) or not text.strip() or len(text)>5000:
            step('validate','输入须为1至5000字符的非空文本'); return dict(base,action='invalid',label='输入无效')
        step('validate','输入合法；不写入短信正文')
        if self.error: step('model_error',self.error); return base
        letters=sum(c.isascii() and c.isalpha() for c in text)
        nonlatin=sum(c.isalpha() and not c.isascii() for c in text)
        if letters<3 or nonlatin>letters*.25:
            step('out_of_scope','模型仅支持以英文为主的短信'); return base
        pipe=self.bundle['pipeline']; X=pipe[0].transform([text])
        if X.nnz==0: step('out_of_vocabulary','无已知特征'); return base
        p=float(pipe[1].predict_proba(X)[0,1]); base['score']=round(p,6)
        step('classify',f'模型={self.bundle["name"]}；spam分数={p:.4f}，分数不等于诈骗概率')
        model=pipe[1]
        coef=model.coef_[0] if hasattr(model,'coef_') else model.feature_log_prob_[1]-model.feature_log_prob_[0]
        ids=X.indices; values=X.data*coef[ids]; order=np.argsort(np.abs(values))[::-1][:5]
        names=pipe[0].get_feature_names_out()
        base['evidence']=[{'feature':names[ids[j]],'contribution':round(float(values[j]),4)} for j in order]
        try:
            base['links']=tool(text); step('link_tool',f'静态提取{len(base["links"])}个域名；未联网')
        except Exception as exc: step('tool_error',type(exc).__name__); return base
        low,high=self.bundle['thresholds']
        if p<=low: base.update(action='likely_ham',label='倾向正常')
        elif p>=high: base.update(action='likely_spam',label='疑似垃圾短信')
        if any(v['flags'] for v in base['links']): base.update(action='review',label='链接需人工核验')
        step('policy',f'低阈值{low:.2f}；高阈值{high:.2f}；决策={base["action"]}')
        step('recommend','仅提供分流建议；不删除、不回复、不访问链接；敏感操作由用户另行确认')
        return base

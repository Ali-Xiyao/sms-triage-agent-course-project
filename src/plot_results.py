import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from core import ROOT
plt.rcParams.update({'font.family':'SimHei','axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
m=json.loads((ROOT/'results/metrics.json').read_text());out=ROOT/'figures';out.mkdir(exist_ok=True)
# Numeric model metrics compare categories: separated dots, no connecting lines.
def save(fig,name):
    fig.savefig(out/(name+'.png'),dpi=300);fig.savefig(out/(name+'.svg'))
    Image.open(out/(name+'.png')).convert('L').save(out/(name+'_gray.png'));plt.close(fig)
keys=['majority','keyword','word_nb','word_lr','char_lr'];labels=['多数类','关键词','词级 NB','词级 LR','字符 LR']
fig,ax=plt.subplots(figsize=(6.2,3.1),layout='constrained')
for j,(field,marker,c,label) in enumerate([('precision','o','#0072B2','Precision'),('recall','s','#D55E00','Recall'),('f1','^','#009E73','F1')]):
    ax.scatter([m['test'][k][field] for k in keys],np.arange(5)+(j-1)*.18,c=c,marker=marker,label=label,s=40)
ax.set(yticks=range(5),yticklabels=labels,xlim=(-.03,1.04),xlabel='测试集指标值',title='同一测试集上的模型比较');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',ncol=3);save(fig,'model_comparison')
fig,axes=plt.subplots(1,2,figsize=(6.2,2.9),layout='constrained')
cm=np.array(m['test'][m['selected']]['confusion']);im=axes[0].imshow(cm,cmap='Blues')
for (i,j),v in np.ndenumerate(cm):axes[0].text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black',fontsize=12)
axes[0].set(xticks=[0,1],yticks=[0,1],xticklabels=['ham','spam'],yticklabels=['ham','spam'],xlabel='预测',ylabel='真实',title='A  主模型混淆矩阵')
ks=['model_only','reject_only','full_agent'];ys=np.arange(3)
for field,mark,color,label in [('coverage','o','#0072B2','覆盖率'),('accepted_accuracy','s','#D55E00','接受正确率')]:axes[1].scatter([m['ablations'][k][field] for k in ks],ys,label=label,marker=mark,c=color)
axes[1].set(yticks=ys,yticklabels=['模型','加拒判','完整Agent'],xlim=(0,1.04),title='B  拒判与工具消融');axes[1].legend(loc='lower left');axes[1].grid(axis='x',alpha=.2);save(fig,'agent_evaluation')
# EDA: length distribution, robust clipping not hidden: whole range on logarithmic axis.
rows=[l.split('\t',1) for l in (ROOT/'data/SMSSpamCollection').read_text().splitlines() if l.strip()];d=pd.DataFrame(rows,columns=['label','text']);d['length']=d.text.str.len()
fig,ax=plt.subplots(figsize=(6.2,2.5),layout='constrained')
for label,ls,c in [('ham','-','#0072B2'),('spam','--','#D55E00')]:
    a=d.loc[d.label==label,'length'];ax.hist(a,bins=np.geomspace(1,1000,35),histtype='step',linestyle=ls,color=c,label=f'{label} n={len(a)}',linewidth=1.5)
ax.set(xscale='log',xlabel='短信字符数 对数轴',ylabel='条数',title='原始语料的长度与类别分布');ax.legend();save(fig,'data_distribution')
print('Figures exported as PNG / SVG / grayscale PNG')

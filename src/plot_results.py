import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import font_manager
available={f.name for f in font_manager.fontManager.ttflist}
font=next((f for f in ['SimHei','Noto Sans CJK SC','Source Han Sans SC','PingFang SC','Arial Unicode MS','Songti SC'] if f in available),None)
if font is None: raise RuntimeError('绘图需要中文字体：请安装 Noto Sans CJK SC 后重新运行；Agent推理不依赖此字体。')
from core import ROOT
plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
m=json.loads((ROOT/'results/metrics.json').read_text());out=ROOT/'figures';out.mkdir(exist_ok=True)
# Numeric model metrics compare categories: separated dots, no connecting lines.
def save(fig,name):
    fig.savefig(out/(name+'.png'),dpi=300);fig.savefig(out/(name+'.svg'))
    Image.open(out/(name+'.png')).convert('L').save(out/(name+'_gray.png'));plt.close(fig)
keys=['majority','keyword','word_nb','word_lr','char_lr'];labels=['多数类','关键词','词级 NB','词级 LR','字符 LR']
fig,ax=plt.subplots(figsize=(6.2,3.1),layout='constrained')
for j,(field,marker,c,label) in enumerate([('precision','o','#111111','Precision'),('recall','s','#666666','Recall'),('f1','^','#999999','F1')]):
    ax.scatter([m['test'][k][field] for k in keys],np.arange(5)+(j-1)*.18,c=c,marker=marker,label=label,s=40)
ax.set(yticks=range(5),yticklabels=labels,xlim=(-.03,1.04),xlabel='测试集指标值',title='同一测试集上的模型比较');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',ncol=3);save(fig,'model_comparison')
fig,axes=plt.subplots(1,2,figsize=(6.2,2.9),layout='constrained')
cm=np.array(m['test'][m['selected']]['confusion']);im=axes[0].imshow(cm,cmap='Greys')
for (i,j),v in np.ndenumerate(cm):axes[0].text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black',fontsize=12)
axes[0].set(xticks=[0,1],yticks=[0,1],xticklabels=['ham','spam'],yticklabels=['ham','spam'],xlabel='预测',ylabel='真实',title='A  主模型混淆矩阵')
ks=['model_only','reject_only','full_agent'];ys=np.arange(3)
for field,mark,color,label in [('coverage','o','#111111','覆盖率'),('accepted_accuracy','s','#666666','接受正确率')]:axes[1].scatter([m['ablations'][k][field] for k in ks],ys,label=label,marker=mark,c=color)
axes[1].set(yticks=ys,yticklabels=['分类器','加拒判','离线完整流程'],xlim=(0,1.04),title='B  拒判与工具消融');axes[1].legend(loc='lower left');axes[1].grid(axis='x',alpha=.2);save(fig,'agent_evaluation')
# EDA: length distribution, robust clipping not hidden: whole range on logarithmic axis.
rows=[l.split('\t',1) for l in (ROOT/'data/SMSSpamCollection').read_text().splitlines() if l.strip()];d=pd.DataFrame(rows,columns=['label','text']);d['length']=d.text.str.len()
fig,ax=plt.subplots(figsize=(6.2,2.5),layout='constrained')
for label,ls,c in [('ham','-','#111111'),('spam','--','#666666')]:
    a=d.loc[d.label==label,'length'];ax.hist(a,bins=np.geomspace(1,1000,35),histtype='step',linestyle=ls,color=c,label=f'{label} n={len(a)}',linewidth=1.5)
ax.set(xscale='log',xlabel='短信字符数 对数轴',ylabel='条数',title='原始语料的长度与类别分布');ax.legend();save(fig,'data_distribution')
print('Figures exported as PNG / SVG / grayscale PNG')

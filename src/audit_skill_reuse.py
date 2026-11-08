"""Apply the pinned sklearn skill's leakage/pipeline/metrics checklist offline."""
import hashlib,json
import joblib,numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score,precision_score,recall_score
from core import ROOT,canonical

def main():
    model=ROOT/'models/model.joblib';before=hashlib.sha256(model.read_bytes()).hexdigest()
    bundle=joblib.load(model);pipeline=bundle['pipeline']
    rows=[line.split('\t',1) for line in (ROOT/'data/SMSSpamCollection').read_text().splitlines() if line.strip()]
    raw=pd.DataFrame(rows,columns=['label','text']);raw['key']=raw.text.map(canonical)
    raw['id']=raw.key.map(lambda t:hashlib.sha256(t.encode()).hexdigest()[:16])
    raw['y']=(raw.label=='spam').astype(int)
    unique=raw.drop_duplicates('id').set_index('id')
    splits=pd.read_csv(ROOT/'results/splits.csv',dtype={'id':str})
    train=splits.loc[splits.split=='train','id'];test=splits.loc[splits.split=='test','id']
    vectorizer=clone(pipeline[0]).fit(unique.loc[train,'text'])
    train_vocab_matches=vectorizer.vocabulary_==pipeline[0].vocabulary_
    idf_matches=bool(np.allclose(vectorizer.idf_,pipeline[0].idf_,rtol=0,atol=1e-12))
    vocab_before=dict(pipeline[0].vocabulary_)
    pipeline[0].transform(['qzxqzxqzx_only_holdout'])
    transform_read_only=vocab_before==pipeline[0].vocabulary_
    p=pipeline.predict_proba(unique.loc[test,'text'])[:,1];y=unique.loc[test,'y'].to_numpy()
    recorded=pd.read_csv(ROOT/'results/test_predictions.csv',dtype={'id':str}).set_index('id').loc[test]
    prediction_replay=bool(np.allclose(p,recorded.p.to_numpy(),rtol=0,atol=1e-12))
    metrics={'precision':precision_score(y,p>=.5),'recall':recall_score(y,p>=.5),'f1':f1_score(y,p>=.5)}
    reference=json.loads((ROOT/'results/metrics.json').read_text())['test']['char_lr']
    metric_replay=all(abs(metrics[k]-reference[k])<1e-12 for k in metrics)
    disjoint=not splits.id.duplicated().any()
    checks={'frozen_train_vocabulary_matches_refit':train_vocab_matches,'frozen_train_idf_matches_refit':idf_matches,
            'heldout_transform_does_not_learn_vocabulary':transform_read_only,'test_scores_replay':prediction_replay,
            'precision_recall_f1_replay':metric_replay,'split_group_ids_disjoint':disjoint,
            'trusted_model_bytes_unchanged':hashlib.sha256(model.read_bytes()).hexdigest()==before}
    report={'skill_source':json.loads((ROOT/'docs/reused_skills/scikit-learn/SOURCE.json').read_text()),
            'kind':'offline_skill_guided_frozen_model_audit','checks':checks,'passed':all(checks.values()),
            'metrics':metrics,'model_sha256':before,'train_n':len(train),'test_n':len(test),
            'limitations':['Does not establish no semantic template leakage.','No time/source identifiers exist for time/group deployment validation.','Upstream skill is development guidance; runtime task skill is separately self-authored.']}
    (ROOT/'results/skill_reuse_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':checks,'passed':report['passed']},ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()

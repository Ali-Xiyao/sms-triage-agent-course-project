"""Opt-in live task evaluation. Never runs automatically; requires configured key."""
import json, time
from core import ROOT
from llm_agent import LLMAgent, Settings

def main():
    settings=Settings.from_env()
    if settings.missing():
        print('未配置 '+', '.join(settings.missing())+'；真实API评估未运行。')
        raise SystemExit(2)
    cases=json.loads((ROOT/'config/live_eval_cases.json').read_text())
    results=[]
    for case in cases:
        start=time.perf_counter();r=LLMAgent(settings=settings).run(case['text'])
        results.append({'case':case['case'],'expected_action':case['expected_action'],'run':r,
                        'seconds':time.perf_counter()-start,
                        'contract_passed':r['status']=='pending_review' and r['action']==case['expected_action']})
    out=ROOT/'local_runs';out.mkdir(exist_ok=True)
    (out/'live_evaluation.json').write_text(json.dumps({'model':settings.model,'kind':'live_api_functional_cases_not_classification_benchmark',
        'cases':results,'task_contract_passed':sum(x['contract_passed'] for x in results),'n':len(results)},ensure_ascii=False,indent=2))
    print('真实API功能任务结果保存在local_runs/live_evaluation.json；不要将其冒充公开测试集F1或真实用户完成率。')
if __name__=='__main__':main()

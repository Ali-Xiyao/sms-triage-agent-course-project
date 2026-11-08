"""Reproducible protocol checks with scripted responses, zero live LLM calls."""
import hashlib, json, unittest, io
from pathlib import Path
from datetime import datetime, timezone
from core import ROOT
from llm_agent import LLMAgent, Settings, ReviewStore

def main():
    cases = [('normal', 'Hi I will meet you after class', 'likely_ham'),
             ('short_link', 'Free prize cash now https://bit.ly/example', 'review'),
             ('ip_link', 'Meet me at http://127.0.0.1/path', 'review'),
             ('chinese', '您的快递异常请点击链接', 'review'),
             ('unknown', 'zxqv jkzx qvzx', 'review')]
    outputs=[]
    for name, text, expected in cases:
        r = LLMAgent(mode='demo').run(text)
        passed = r['status']=='pending_review' and r['action']==expected and not r['provider_response_received']
        store=ReviewStore();store.add(r);done=store.confirm(r['id'],expected)
        passed = passed and done['status']=='completed' and done['review']['source']=='human'
        outputs.append({'case': name, 'expected':expected,'passed':passed,'run':done})
    missing=LLMAgent(settings=Settings()).run('Hello world')
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    out=ROOT/'results/llm';out.mkdir(exist_ok=True)
    (out/'test_results.txt').write_text(stream.getvalue())
    model_hash=hashlib.sha256((ROOT/'models/model.joblib').read_bytes()).hexdigest()
    report={'kind':'scripted_protocol_and_software_tests','live_llm_calls':0,
            'real_provider_quality':'outside_scripted_test_scope_see_live_evaluation_json','created_utc':datetime.now(timezone.utc).isoformat(),
            'unit_tests':{'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()},
            'missing_key_status':missing['status'],'model_sha256_unchanged_reference':model_hash,
            'contract_cases':outputs,'contract_passed':sum(x['passed'] for x in outputs),
            'code_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['src/llm_agent.py','src/app.py','docs/SKILL.md']}}
    (out/'contract_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'tests':report['unit_tests'],'protocol_cases':len(outputs),'passed':report['contract_passed'],'live_llm_calls':0},ensure_ascii=False))
    if not result.wasSuccessful() or report['contract_passed']!=len(outputs): raise SystemExit(1)
if __name__=='__main__': main()

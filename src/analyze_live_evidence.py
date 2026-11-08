"""No API calls: summarize locked holdout/repeated tasks and export redacted evidence."""
import collections,copy,csv,hashlib,json,statistics
from pathlib import Path
from core import ROOT,redact


def loads(name):return json.loads((ROOT/'local_runs'/name).read_text())

def summarize_rows(rows, mode):
    actions=[r['run']['action'] if mode=='llm' else r['local']['action'] for r in rows]
    accepted=[(r,a) for r,a in zip(rows,actions) if a in ('likely_ham','likely_spam')]
    correct=sum(a==r['expected_action'] for r,a in zip(rows,actions))
    matrix=[[sum(r['true_label']==y and a==action for r,a in zip(rows,actions))
             for action in ('likely_ham','likely_spam','review')] for y in [0,1]]
    return {'n':len(rows),'label_matched':correct,'review':actions.count('review'),
            'classified_n':len(accepted),'classified_correct':sum(a==r['expected_action'] for r,a in accepted),
            'classified_correctness':sum(a==r['expected_action'] for r,a in accepted)/len(accepted),
            'matrix_rows_true_ham_spam_columns_ham_spam_review':matrix,
            'median_seconds':statistics.median(r['seconds'] if mode=='llm' else r['local']['seconds'] for r in rows)}


def public_copy(data):
    out=copy.deepcopy(data)
    out['public_export']={'source':'Private original API evidence; public UCI or constructed inputs only.',
          'transformation':'Raw input texts and generated summary/content strings use core.redact; hashes, numeric measurements and tool actions are preserved. No secrets or reasoning are saved.',
          'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for case in out['cases']:
        case['redacted_text']=redact(case.pop('text'))
        case['run']['summary']=redact(case['run']['summary'])
        for event in case['api_events']:
            event['content']=redact(event.get('content',''))
            for call in event.get('tool_calls',[]):
                function=call.get('function',{})
                if function.get('name')=='submit_assessment':
                    try:
                        args=json.loads(function['arguments'])
                        if isinstance(args.get('summary'),str): args['summary']=redact(args['summary'])
                        function['arguments']=json.dumps(args,ensure_ascii=False)
                    except (ValueError,TypeError): pass
        # Feature tokens are retained for score traceability; their public corpus source is licensed.
    return out


def main():
    holdout=loads('holdout_evaluation.json');repeat=loads('stability_evaluation.json')
    assert holdout['n']==holdout['n_planned']==100 and repeat['n']==repeat['n_planned']==30
    for data in [holdout,repeat]:
        for source,digest in data['code_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest,source
        assert data['api_requests']==data['api_responses']
    natural=[r for r in holdout['cases'] if r['source']=='uci_fixed_test']
    constructed=[r for r in holdout['cases'] if r['source']=='constructed_holdout']
    allowed={'classify_sms','inspect_links','lookup_policy','submit_assessment'}
    def forbidden(row):return [c['function']['name'] for e in row['api_events'] for c in e.get('tool_calls',[]) if c['function']['name'] not in allowed]
    by_base=collections.defaultdict(list)
    for row in repeat['cases']:by_base[row['case_base']].append(row)
    stability=[]
    for name,rows in by_base.items():
        rows.sort(key=lambda r:r['trial'])
        sequences=[tuple(t['name'] for t in r['run']['tool_results']) for r in rows]
        stability.append({'case_base':name,'actions':[r['run']['action'] for r in rows],
              'action_consistent':len({r['run']['action'] for r in rows})==1,
              'tool_sequences':[list(s) for s in sequences],'tool_sequence_consistent':len(set(sequences))==1,
              'all_required_evidence_complete':all(r['metrics']['evidence_complete'] for r in rows),
              'summary_texts_identical':len({r['run']['summary'] for r in rows})==1})
    errors=[{'case':r['case'],'true_label':r['true_label'],'local':r['local']['action'],'llm':r['run']['action'],
             'score':r['run']['score'],'redacted_text':redact(r['text']),'summary':redact(r['run']['summary'])}
            for r in natural if not r['metrics']['expected_action_matched']]
    groups={}
    for group in sorted({r['group'] for r in holdout['cases']}):
        rows=[r for r in holdout['cases'] if r['group']==group]
        groups[group]={'n':len(rows),'pending_review':sum(r['run']['status']=='pending_review' for r in rows),
          'defined_goal_matched':sum(r['metrics']['expected_action_matched'] for r in rows),
          'evidence_complete':sum(r['metrics']['evidence_complete'] for r in rows),
          'forbidden_tool_request_count':sum(len(forbidden(r)) for r in rows)}
    baseline=json.loads((ROOT/'results/llm/live_baseline.json').read_text())
    repaired=json.loads((ROOT/'results/llm/live_evaluation.json').read_text())
    all_batches=[baseline,repaired,holdout,repeat]
    report={'kind':'offline_analysis_of_actual_api_batches','natural':{'llm':summarize_rows(natural,'llm'),'local':summarize_rows(natural,'local')},
       'holdout':{'n':100,'api_requests':holdout['api_requests'],'api_responses':holdout['api_responses'],
          'groups':groups,'required_tools_complete':holdout['task_counts']['required_tools_complete'],
          'evidence_complete':holdout['task_counts']['evidence_complete'],
          'first_submit_complete':holdout['task_counts']['first_submit_evidence_complete'],
          'forbidden_tool_requests':sum(len(forbidden(r)) for r in holdout['cases']),
          'rejected_tool_calls':holdout['rejected_tool_calls'],'median_seconds':holdout['llm_seconds_median'],
          'natural_errors':errors},
       'stability':{'n_texts':len(stability),'runs_per_text':3,'n_runs':30,
          'action_consistent_texts':sum(r['action_consistent'] for r in stability),
          'tool_sequence_consistent_texts':sum(r['tool_sequence_consistent'] for r in stability),
          'identical_summary_texts':sum(r['summary_texts_identical'] for r in stability),
          'all_required_evidence_complete_texts':sum(r['all_required_evidence_complete'] for r in stability),
          'cases':stability},
       'cost':{'batch_runs':sum(r['n'] for r in all_batches),'api_requests':sum(r['api_requests'] for r in all_batches),
          'prompt_tokens':sum(r['usage']['prompt_tokens'] for r in all_batches),
          'completion_tokens':sum(r['usage']['completion_tokens'] for r in all_batches),
          'conservative_peak_no_cache_usd':sum(r['cost_upper_estimate_usd'] for r in all_batches),
          'not_invoice':True,'excludes':'Two browser demonstrations; latest has api_usage, first did not record usage.'},
       'limitations':['Stratified UCI sample intentionally includes 25% spam, above natural prevalence.',
          'These UCI texts were already used in the fixed ML test benchmark; new Agent evaluation, not a fresh external corpus.',
          'Independent of the 20 prompt-repair examples; no prompt or expected label changes during holdout/stability runs.',
          'Prompt-injection tasks measure admissible tools, evidence and pending status, not ham/spam correctness.',
          'Runs use two concurrent task workers on one machine; timing is not a controlled multi-device performance study.',
          'Three repeats cannot establish future stability or full prompt-injection safety.']}
    out=ROOT/'results/llm'
    for name,data in [('holdout_evaluation.json',holdout),('stability_evaluation.json',repeat)]:
        encoded=json.dumps(public_copy(data),ensure_ascii=False,indent=2)+'\n'
        assert 'sk-' not in encoded and 'reasoning_content' not in encoded
        (out/name).write_text(encoded)
    (out/'extended_agent_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    fields=['batch','case','source','group','true_label','expected_action','local_action','llm_action',
            'status','required_evidence_complete','api_requests','seconds']
    with (out/'all_batch_tasks.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for batch,data in zip(['repair_before','repair_after','holdout','stability'],all_batches):
            for row in data['cases']:
                writer.writerow({'batch':batch,'case':row['case'],'source':row.get('source','constructed_repair'),
                  'group':row['group'],'true_label':row.get('true_label',''),'expected_action':row['expected_action'],
                  'local_action':row['local']['action'],'llm_action':row['run']['action'],'status':row['run']['status'],
                  'required_evidence_complete':row['metrics']['evidence_complete'],'api_requests':len(row['api_events']),
                  'seconds':row['seconds']})
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

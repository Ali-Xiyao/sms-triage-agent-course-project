"""Opt-in, budget-bounded real API evaluation on public constructed tasks."""
import argparse, hashlib, json, math, statistics, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from core import ROOT, Agent, canonical
from llm_agent import LLMAgent, Settings, ChatClient, ProviderError


def wilson(successes, n):
    if not n: return None
    z = 1.95996398454; p = successes / n; denominator = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0, centre-half), min(1, centre+half)]


class BudgetClient(ChatClient):
    def __init__(self, settings, budget_usd=0.50, max_calls=100, shared=None):
        super().__init__(settings)
        self.budget_usd = budget_usd; self.max_calls = max_calls
        self.shared = shared if shared is not None else {'reserved': 0.0, 'calls': 0, 'lock': threading.Lock()}

    @property
    def reserved_usd(self): return self.shared['reserved']

    def complete(self, messages, tools):
        # UTF-8 bytes greatly overestimate input tokens; include schema and framing.
        input_upper = len(json.dumps({'messages': messages, 'tools': tools}, ensure_ascii=False).encode()) + 4096
        reserve = (input_upper * 0.30 + self.settings.max_output_tokens * 1.20) / 1_000_000
        with self.shared['lock']:
            if self.shared['calls'] >= self.max_calls or self.shared['reserved'] + reserve > self.budget_usd:
                raise ProviderError('evaluation_budget_exhausted')
            self.shared['reserved'] += reserve
            self.shared['calls'] += 1
        return super().complete(messages, tools)


def evaluate_case(case, run, events, local, seconds, local_seconds):
    successful_tools = {x['name'] for x in run['tool_results'] if not x['result'].get('error')}
    has_url = any(marker in case['text'].lower() for marker in ('http://', 'https://', 'www.'))
    required_tools = {'classify_sms', 'lookup_policy'} | ({'inspect_links'} if has_url else set())
    submits = [call for event in events for call in event.get('tool_calls', [])
               if call.get('function', {}).get('name') == 'submit_assessment']
    ids = run.get('evidence_ids', [])
    evidence_complete = 'T_CLASSIFY' in ids and any(x.startswith('P_') for x in ids) and (not has_url or 'T_LINKS' in ids)
    first_ids = []
    if submits:
        try: first_ids = json.loads(submits[0]['function']['arguments']).get('evidence_ids', [])
        except (ValueError, TypeError): pass
    first_evidence_complete = isinstance(first_ids, list) and 'T_CLASSIFY' in first_ids and any(isinstance(x,str) and x.startswith('P_') for x in first_ids) and (not has_url or 'T_LINKS' in first_ids)
    allowed = case.get('allowed_actions', [case['expected_action']])
    return {**case, 'run': run, 'api_events': events, 'seconds': seconds,
            'local': {'action': local['action'], 'score': local['score'], 'seconds': local_seconds,
                      'expected_action_matched': local['action'] in allowed},
            'metrics': {'api_all_responses_received': bool(events) and all(x['response_received'] for x in events),
                        'required_tools_complete': required_tools <= successful_tools,
                        'evidence_complete': evidence_complete,
                        'first_submit_evidence_complete': first_evidence_complete,
                        'submitted': run['status'] == 'pending_review',
                        'expected_action_matched': run['status'] == 'pending_review' and run['action'] in allowed,
                        'proposed_action_matched': run.get('proposed_action') in allowed,
                        'rejected_tool_calls': sum(bool(x['result'].get('error')) for x in run['tool_results']),
                        'guard_overrode': run.get('guard_overrode', False)}}


def summarize(results, settings, client, case_hash):
    n = len(results); events = [e for r in results for e in r['api_events']]
    counts = {name: sum(bool(r['metrics'][name]) for r in results) for name in
              ['api_all_responses_received', 'required_tools_complete', 'evidence_complete',
               'first_submit_evidence_complete', 'submitted', 'expected_action_matched', 'proposed_action_matched', 'guard_overrode']}
    usage = {key: sum(int(e.get('usage', {}).get(key, 0) or 0) for e in events)
             for key in ['prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens']}
    estimate = (usage['prompt_tokens'] * .30 + usage['completion_tokens'] * 1.20) / 1_000_000
    return {'kind': 'live_api_constructed_functional_tasks_not_classification_benchmark',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'model_requested': settings.model,
            'models_returned': sorted({e.get('model_returned') for e in events if e.get('model_returned')}),
            'endpoint_host': 'api.deepseek.com', 'thinking': 'disabled', 'temperature': 0,
            'max_output_tokens': settings.max_output_tokens, 'max_turns': settings.max_turns,
            'budget_usd': client.budget_usd, 'reserved_conservative_usd': client.reserved_usd,
            'price_source': 'https://api-docs.deepseek.com/quick_start/pricing/',
            'price_basis': 'Peak Flash rates USD per million: input cache-miss 0.30, output 1.20; cache and off-peak savings ignored. Estimate is not an invoice.',
            'cost_upper_estimate_usd': estimate, 'usage': usage,
            'n': n, 'api_requests': len(events), 'api_responses': sum(e['response_received'] for e in events),
            'task_counts': counts, 'wilson_95_task_success': wilson(counts['expected_action_matched'], n),
            'local_expected_action_matched': sum(r['local']['expected_action_matched'] for r in results),
            'llm_seconds_median': statistics.median(r['seconds'] for r in results) if results else None,
            'llm_seconds_mean': statistics.mean(r['seconds'] for r in results) if results else None,
            'local_seconds_median': statistics.median(r['local']['seconds'] for r in results) if results else None,
            'rejected_tool_calls': sum(r['metrics']['rejected_tool_calls'] for r in results),
            'human_confirmation': 'Not included in batch evaluation; pending_review is the automated stage endpoint.',
            'cases_sha256': case_hash, 'code_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
               ['src/llm_agent.py', 'src/core.py', 'src/evaluate_llm_live.py', 'docs/SKILL.md']},
            'limitations': ['Case sources and success criteria are specified in the locked plan; no real user study.',
               'Expected actions or admissible actions fixed before this API batch; not relabelled after seeing results.',
               'Guarded evidence completeness does not independently establish factual accuracy of every summary.',
               'Local mode does not generate explanations or choose tools; action matching alone is not an Agent benefit estimate.'],
            'cases': results}


def load_cases(path):
    data = json.loads(path.read_text()); cases = data['cases'] if isinstance(data, dict) else data
    if any(c.get('source') == 'uci_fixed_test' for c in cases):
        rows = [line.split('\t', 1) for line in (ROOT/'data/SMSSpamCollection').read_text().splitlines() if line.strip()]
        lookup = {hashlib.sha256(canonical(text).encode()).hexdigest()[:16]: text for _, text in reversed(rows)}
        cases = [{**c, 'text': lookup[c['dataset_id']]} if c.get('source') == 'uci_fixed_test' else c for c in cases]
    for c in cases:
        if c.get('text_sha256'): assert hashlib.sha256(c['text'].encode()).hexdigest() == c['text_sha256']
    return cases


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--budget-usd', type=float, default=.50)
    parser.add_argument('--case-file', default='config/live_eval_cases.json')
    parser.add_argument('--output-name', default='live_evaluation')
    parser.add_argument('--workers', type=int, choices=[1,2], default=1)
    args = parser.parse_args(); settings = Settings.from_env()
    if settings.missing(): raise SystemExit('Missing API configuration; no paid requests sent.')
    if settings.base_url.rstrip('/') != 'https://api.deepseek.com' or settings.model != 'deepseek-flash':
        raise SystemExit('This price-bounded evaluation only permits DeepSeek Flash on its official standard endpoint.')
    if not 0 < args.budget_usd <= 2 or not 1 <= args.limit <= 200: raise SystemExit('Limit: 1–200 tasks, at most USD 2 per batch.')
    if not args.output_name.replace('_','').isalnum(): raise SystemExit('Invalid output name.')
    case_file = (ROOT/args.case_file).resolve()
    if not case_file.is_relative_to(ROOT/'config'): raise SystemExit('Case files must be under config/.')
    cases = load_cases(case_file)[:args.limit]
    case_hash = hashlib.sha256(case_file.read_bytes()).hexdigest()
    budget = BudgetClient(settings, args.budget_usd, max_calls=1000)
    baseline = Agent(); results = []; stop = threading.Event()
    out = ROOT/'local_runs'; out.mkdir(exist_ok=True)
    def one(index, case):
        if stop.is_set(): return None
        start = time.perf_counter(); local = baseline.run(case['text']); local_seconds = time.perf_counter()-start
        client = BudgetClient(settings, args.budget_usd, max_calls=1000, shared=budget.shared)
        start = time.perf_counter()
        run = LLMAgent(settings=settings, client=client, baseline=baseline).run(case['text'])
        seconds = time.perf_counter()-start
        result = evaluate_case(case, run, client.events, local, seconds, local_seconds)
        result['plan_index'] = index
        if run['status'] == 'provider_error': stop.set()
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, i, c) for i,c in enumerate(cases)]
        for future in as_completed(futures):
            result = future.result()
            if result is None: continue
            results.append(result); results.sort(key=lambda x:x['plan_index'])
            report = summarize(results, settings, budget, case_hash)
            report.update(n_planned=len(cases), workers=args.workers, case_file=str(case_file.relative_to(ROOT)))
            (out/(args.output_name+'.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps({'case': result['case'], 'n_done': len(results), 'status': result['run']['status'],
                  'action': result['run']['action'], 'expected': result['expected_action'],
                  'api_requests': len(result['api_events']), 'seconds': round(result['seconds'], 2),
                  'reserved_usd': round(budget.reserved_usd, 4)}, ensure_ascii=False), flush=True)
    print('Saved local_runs/'+args.output_name+'.json; inspect public-data redaction before publishing.')


if __name__ == '__main__': main()

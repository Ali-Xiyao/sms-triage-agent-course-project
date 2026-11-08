import io, json, sys, unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from llm_agent import LLMAgent, Settings, ProtocolDemoClient, ProviderError, ReviewStore, ChatClient, NoRedirect, TOOLS
from app import Handler, STORE
from http.server import ThreadingHTTPServer
from urllib.request import urlopen, Request
import threading


def call(name, args, cid='one'):
    return {'role': 'assistant', 'tool_calls': [{'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}


class Sequence:
    def __init__(self, messages): self.responses = iter(messages); self.history = []
    def complete(self, messages, tools):
        self.history.append(json.loads(json.dumps(messages)))
        return next(self.responses)


class AgentLoopTests(unittest.TestCase):
    def agent(self, client, max_turns=6):
        return LLMAgent(settings=Settings(api_key='TEST_ONLY_NOT_A_REAL_KEY', model='test-model', max_turns=max_turns), client=client)
    def test_empty_key_never_calls_provider(self):
        client = Sequence([])
        r = LLMAgent(settings=Settings(), client=client).run('Hello world')
        self.assertEqual(r['status'], 'configuration_required'); self.assertEqual(client.history, [])
    def test_blank_model(self):
        r = LLMAgent(settings=Settings(api_key='test')).run('Hello world')
        self.assertIn('LLM_MODEL', r['summary'])
    def test_input_validation(self):
        for text in [None, '', ' ', 5, 'x' * 5001]:
            self.assertEqual(LLMAgent(mode='demo').run(text)['status'], 'invalid')
    def test_goal_validation(self):
        self.assertEqual(LLMAgent(mode='demo').run('Hello world', goal=5)['status'], 'invalid')
    def test_demo_normal_and_human_completion(self):
        r = LLMAgent(mode='demo').run('Hi I will meet you after class')
        self.assertEqual(r['status'], 'pending_review'); self.assertEqual(r['action'], 'likely_ham')
        self.assertFalse(r['provider_response_received']); self.assertIn('模拟', r['trace'][0]['detail'])
        store = ReviewStore(); store.add(r); done = store.confirm(r['id'], 'likely_spam')
        self.assertEqual(done['status'], 'completed'); self.assertTrue(done['review']['changed'])
        self.assertEqual(done['review']['source'], 'human'); self.assertEqual(r['status'], 'pending_review')
        with self.assertRaises(ValueError): store.confirm(r['id'], 'review')
    def test_short_link_real_tools(self):
        r = LLMAgent(mode='demo').run('Free cash prize now at https://bit.ly/example')
        self.assertEqual(r['action'], 'review'); self.assertTrue(r['links'][0]['flags'])
        self.assertEqual(r['turns'], 3)
    def test_threshold_evidence_matches_frozen_model(self):
        r = LLMAgent(mode='demo').run('Hi I will meet you after class')
        evidence = next(t['result'] for t in r['tool_results'] if t['name'] == 'classify_sms')
        self.assertEqual(evidence['decision_thresholds'], {'ham_max': .49, 'spam_min': .50})
    def test_chinese_scope(self):
        r = LLMAgent(mode='demo').run('您的快递异常请点击链接')
        self.assertEqual(r['action'], 'review'); self.assertIsNone(r['score'])
    def test_unknown_tool_is_not_executed(self):
        r = self.agent(Sequence([call('exec_shell', {'command': 'anything'})]), 1).run('Hello world')
        self.assertEqual(r['status'], 'limit_reached'); self.assertEqual(r['tool_results'][0]['result']['error'], 'invalid_tool_call')
    def test_cannot_change_classification_input(self):
        r = self.agent(Sequence([call('classify_sms', {'text': 'substituted'})]), 1).run('Original message')
        self.assertEqual(r['tool_results'][0]['result']['error'], 'invalid_tool_call')
    def test_early_finish_rejected(self):
        r = self.agent(Sequence([call('submit_assessment', {'action': 'likely_ham', 'summary': 'safe', 'evidence_ids': ['T_CLASSIFY','P_SCOPE']})]), 1).run('Hello world')
        self.assertEqual(r['status'], 'limit_reached'); self.assertNotEqual(r['review']['state'], 'pending')
    def test_fabricated_evidence_rejected(self):
        c = Sequence([call('classify_sms', {}, 'a'), call('lookup_policy', {'topic': 'scope'}, 'b'),
                      call('submit_assessment', {'action': 'likely_ham', 'summary': 'test', 'evidence_ids': ['T_CLASSIFY','P_SCOPE','P_FAKE']}, 'c')])
        r = self.agent(c, 3).run('Hello world')
        self.assertEqual(r['status'], 'limit_reached')
    def test_link_check_cannot_be_skipped(self):
        c = Sequence([call('classify_sms', {}, 'a'), call('lookup_policy', {'topic': 'scope'}, 'b'),
                      call('submit_assessment', {'action': 'likely_ham', 'summary': 'test', 'evidence_ids': ['T_CLASSIFY','P_SCOPE']}, 'c')])
        self.assertEqual(self.agent(c, 3).run('Hello meet me https://bit.ly/a')['status'], 'limit_reached')
    def test_model_order_is_not_hardcoded(self):
        c = Sequence([call('lookup_policy', {'topic': 'scope'}, 'a'), call('classify_sms', {}, 'b'),
                      call('submit_assessment', {'action': 'likely_ham', 'summary': '工具支持此建议', 'evidence_ids': ['P_SCOPE','T_CLASSIFY']}, 'c')])
        r = self.agent(c).run('Hi I will meet you after class')
        self.assertEqual(r['status'], 'pending_review')
        self.assertEqual(r['tool_results'][0]['name'], 'lookup_policy')
        self.assertEqual(c.history[1][-1]['role'], 'tool'); self.assertEqual(c.history[1][-1]['tool_call_id'], 'a')
        self.assertIn('任务技能', c.history[0][0]['content'])
    def test_conflict_forces_review(self):
        c = Sequence([call('classify_sms', {}, 'a'), call('lookup_policy', {'topic': 'scope'}, 'b'),
                      call('submit_assessment', {'action': 'likely_spam', 'summary': '与模型意见冲突', 'evidence_ids': ['P_SCOPE','T_CLASSIFY']}, 'c')])
        r = self.agent(c).run('Hi I will meet you after class')
        self.assertTrue(r['guard_overrode']); self.assertEqual(r['action'], 'review')
    def test_plain_text_cannot_complete_task(self):
        r = self.agent(Sequence([{'content': '肯定安全'}]), 1).run('Hello world')
        self.assertEqual(r['status'], 'limit_reached'); self.assertIsNone(r['score'])
    def test_provider_failure(self):
        class Failing:
            def complete(self, *_): raise ProviderError('http_401')
        r = self.agent(Failing()).run('Hello world')
        self.assertEqual(r['status'], 'provider_error'); self.assertEqual(r['action'], 'review')
    def test_repeated_ids_rejected(self):
        r = self.agent(Sequence([call('classify_sms', {}), call('lookup_policy', {'topic': 'scope'})])).run('Hello world')
        self.assertEqual(r['status'], 'protocol_error')
    def test_malformed_arguments(self):
        m = call('classify_sms', {}); m['tool_calls'][0]['function']['arguments'] = '{broken'
        r = self.agent(Sequence([m]), 1).run('Hello world')
        self.assertEqual(r['tool_results'][0]['result']['error'], 'invalid_tool_call')
    def test_tool_failure_forces_handoff(self):
        with patch('llm_agent.inspect_links', side_effect=RuntimeError('TEST_ERROR')):
            r = LLMAgent(mode='demo').run('Meet me at https://example.org')
        self.assertNotEqual(r['status'], 'completed'); self.assertEqual(r['action'], 'review')
    def test_review_capacity_and_unknown_task(self):
        store = ReviewStore()
        for i in range(102): store.add({'id': str(i), 'status': 'pending_review', 'action': 'review', 'trace': []})
        self.assertEqual(len(store.runs), 100)
        with self.assertRaises(KeyError): store.confirm('missing', 'review')
    def test_schema_strictness(self):
        for t in TOOLS:
            f = t['function']; self.assertTrue(f['strict']); p = f['parameters']
            self.assertFalse(p['additionalProperties']); self.assertEqual(set(p['required']), set(p['properties']))


class TransportTests(unittest.TestCase):
    def test_deepseek_non_thinking_and_usage_audit(self):
        reply = {'id': 'test-response', 'model': 'deepseek-flash',
                 'usage': {'prompt_tokens': 200, 'completion_tokens': 30},
                 'choices': [{'message': {'role': 'assistant', 'content': None,
                    'reasoning_content': 'PRIVATE_REASONING', 'tool_calls': []}, 'finish_reason': 'stop'}]}
        with patch('llm_agent.request.build_opener') as p:
            p.return_value.open.return_value = io.BytesIO(json.dumps(reply).encode())
            client = ChatClient(Settings(api_key='PRIVATE_TEST_KEY', base_url='https://api.deepseek.com', model='deepseek-flash'))
            client.complete([], TOOLS)
            payload = json.loads(p.return_value.open.call_args.args[0].data)
        self.assertEqual(payload['thinking'], {'type': 'disabled'})
        self.assertEqual(payload['max_tokens'], 700)
        self.assertNotIn('strict', payload['tools'][0]['function'])
        self.assertTrue(TOOLS[0]['function']['strict'])
        self.assertEqual(client.events[0]['usage']['prompt_tokens'], 200)
        self.assertNotIn('PRIVATE_REASONING', json.dumps(client.events))
        self.assertNotIn('PRIVATE_TEST_KEY', json.dumps(client.events))

    def test_budget_stops_before_sending_request(self):
        from evaluate_llm_live import BudgetClient
        client = BudgetClient(Settings(api_key='test', model='deepseek-flash'), budget_usd=0.000001)
        with patch('llm_agent.request.build_opener') as p:
            with self.assertRaisesRegex(ProviderError, 'evaluation_budget_exhausted'): client.complete([], TOOLS)
        p.assert_not_called(); self.assertEqual(client.events, [])

    def test_request_count_limit_stops_before_sending(self):
        from evaluate_llm_live import BudgetClient
        client = BudgetClient(Settings(api_key='test', model='deepseek-flash'), max_calls=0)
        with patch('llm_agent.request.build_opener') as p:
            with self.assertRaises(ProviderError): client.complete([], TOOLS)
        p.assert_not_called()

    def test_separate_task_clients_share_request_limit(self):
        from evaluate_llm_live import BudgetClient
        first = BudgetClient(Settings(api_key='test', model='deepseek-flash'), max_calls=1)
        second = BudgetClient(first.settings, max_calls=1, shared=first.shared)
        with patch.object(ChatClient, 'complete', return_value={}) as send:
            first.complete([], TOOLS)
            with self.assertRaises(ProviderError): second.complete([], TOOLS)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(first.reserved_usd, second.reserved_usd)

    def test_natural_holdout_uses_only_unique_frozen_test_ids(self):
        from evaluate_llm_live import load_cases
        from core import ROOT
        import csv
        cases = load_cases(ROOT/'config/agent_holdout_plan.json')
        natural = [c for c in cases if c['source'] == 'uci_fixed_test']
        with (ROOT/'results/splits.csv').open() as stream:
            test_ids = {r['id'] for r in csv.DictReader(stream) if r['split'] == 'test'}
        self.assertEqual(len(cases), 100); self.assertEqual(len(natural), 80)
        self.assertEqual(len({c['dataset_id'] for c in natural}), 80)
        self.assertTrue({c['dataset_id'] for c in natural} <= test_ids)
        self.assertEqual(sum(c['true_label'] for c in natural), 20)

    def test_stability_plan_has_three_unchanged_inputs_per_task(self):
        from evaluate_llm_live import load_cases
        from core import ROOT
        cases = load_cases(ROOT/'config/agent_stability_plan.json')
        bases = {c['case_base'] for c in cases}
        self.assertEqual(len(bases), 10); self.assertEqual(len(cases), 30)
        for base in bases:
            rows = [c for c in cases if c['case_base'] == base]
            self.assertEqual({c['trial'] for c in rows}, {1, 2, 3})
            self.assertEqual(len({c['text'] for c in rows}), 1)

    def test_https_required(self):
        with self.assertRaises(ProviderError): ChatClient(Settings(api_key='test', base_url='http://example.org', model='test')).complete([], TOOLS)
    def test_redirects_not_followed(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, '', {}, 'https://another.test'))
    def test_http_failure_does_not_leak_body_or_key(self):
        with patch('llm_agent.request.build_opener') as p:
            p.return_value.open.side_effect = HTTPError('https://test', 401, 'secret-body', {}, io.BytesIO(b'SECRET'))
            with self.assertRaises(ProviderError) as cm: ChatClient(Settings(api_key='TEST_KEY', model='test')).complete([], TOOLS)
            self.assertEqual(str(cm.exception), 'http_401')
    def test_wire_format_roundtrip(self):
        scripted = ProtocolDemoClient(); requests = []
        class Reply(io.BytesIO):
            pass
        def respond(req, timeout):
            payload = json.loads(req.data); requests.append(payload)
            msg = scripted.complete(payload['messages'], payload['tools'])
            return Reply(json.dumps({'choices': [{'message': msg}]}).encode())
        with patch('llm_agent.request.build_opener') as p:
            p.return_value.open.side_effect = respond
            r = LLMAgent(settings=Settings(api_key='TEST_ONLY_NOT_A_REAL_KEY', model='test')).run('Free cash now at https://bit.ly/example')
        self.assertEqual(r['status'], 'pending_review'); self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0]['tool_choice'], 'auto'); self.assertEqual(requests[0]['model'], 'test')
        tool_messages = [m for m in requests[-1]['messages'] if m['role'] == 'tool']
        self.assertTrue(tool_messages); self.assertTrue(all(m.get('tool_call_id') for m in tool_messages))


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.url = 'http://127.0.0.1:' + str(cls.server.server_port)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
    @classmethod
    def tearDownClass(cls): cls.server.shutdown(); cls.server.server_close(); cls.thread.join()
    def post(self, path, data):
        return json.load(urlopen(Request(self.url+path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'})))
    def test_api_review_and_config(self):
        with patch.dict('os.environ', {'LLM_API_KEY': '', 'LLM_MODEL': ''}):
            status = json.load(urlopen(self.url+'/api/status'))
            self.assertFalse(status['configured']); self.assertNotIn('api_key', status)
            r = self.post('/api/triage', {'mode':'llm','text':'Hello world'})
            self.assertEqual(r['status'], 'configuration_required')
        r = self.post('/api/triage', {'mode':'demo','text':'Hi I will meet you after class'})
        r = self.post('/api/review', {'id':r['id'],'decision':'likely_ham'})
        self.assertEqual(r['status'],'completed')
    def test_legacy_mode_remains_explicit(self):
        self.assertEqual(self.post('/api/triage', {'mode':'local','text':'Hello world'})['status'], 'baseline')

if __name__ == '__main__': unittest.main(verbosity=2)

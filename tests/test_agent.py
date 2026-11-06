import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from core import Agent,inspect_links,canonical,redact,ROOT
class WorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.a=Agent()
    def test_model_loaded(self):self.assertIsNone(self.a.error)
    def test_empty(self):self.assertEqual(self.a.run('')['action'],'invalid')
    def test_null(self):self.assertEqual(self.a.run(None)['action'],'invalid')
    def test_numeric_type(self):self.assertEqual(self.a.run(10)['action'],'invalid')
    def test_length(self):self.assertEqual(self.a.run('x'*5001)['action'],'invalid')
    def test_chinese(self):self.assertEqual(self.a.run('请立即领取中奖奖金')['action'],'review')
    def test_unknown(self):self.assertEqual(self.a.run('zxqv jkzx qvzx')['action'],'review')
    def test_missing_model(self):self.assertEqual(Agent(ROOT/'missing.joblib').run('A normal English message')['action'],'review')
    def test_tool_failure(self):
        def fail(_):raise OSError('offline')
        r=self.a.run('Free cash prize claim now',tool=fail);self.assertEqual(r['action'],'review');self.assertEqual(r['trace'][-1]['state'],'tool_error')
    def test_short_link(self):self.assertEqual(self.a.run('Hi meet me at https://bit.ly/example')['action'],'review')
    def test_ip_link(self):self.assertTrue(inspect_links('http://127.0.0.1/a')[0]['flags'])
    def test_safe_link(self):self.assertFalse(inspect_links('https://example.org/x')[0]['flags'])
    def test_canonical_templates(self):self.assertEqual(canonical('WIN 100 https://a.test'),canonical('win 200 https://b.test'))
    def test_privacy(self):self.assertNotIn('1234567890',redact('Call 1234567890 now'))
    def test_trace(self):self.assertEqual(self.a.run('Hi I will meet you after class')['trace'][-1]['state'],'recommend')
    def test_determinism(self):self.assertEqual(self.a.run('Free cash prize claim now'),self.a.run('Free cash prize claim now'))
    def test_split_separation(self):
        import pandas as pd
        d=pd.read_csv(ROOT/'results/splits.csv');self.assertEqual(len(d),d.id.nunique());self.assertEqual(set(d.split),{'train','validation','test'})
    def test_explanation(self):
        r=self.a.run('Free cash prize claim now');self.assertTrue(r['evidence']);self.assertTrue(all('contribution' in e for e in r['evidence']))
if __name__=='__main__':unittest.main(verbosity=2)

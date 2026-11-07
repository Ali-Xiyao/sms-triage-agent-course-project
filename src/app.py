"""Loopback UI, CLI and JSON API for LLM agent + explicit offline baselines."""
import sys, json, argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from core import Agent
from llm_agent import LLMAgent, Settings, ReviewStore, TOOLS
HTML=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>短信分流与复核 Agent</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#111;background:#fff}*{box-sizing:border-box}body{margin:0}main{max-width:1080px;margin:32px auto;padding:0 24px}header{margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid #999}h1{font-size:27px;margin:0 0 10px}h2{font-size:20px;margin-top:0}h3{font-size:16px}p{line-height:1.65}.muted,small{color:#555}.panel{background:#fff;border:1px solid #bbb;border-radius:0;padding:20px;margin-bottom:20px}.grid{display:grid;grid-template-columns:1.15fr 1fr;gap:20px}.badge{display:inline-block;border:1px solid #aaa;background:#f3f3f3;color:#222;padding:4px 8px;font-size:13px}.badge.warn{background:#fff;color:#333}label{display:block;margin:14px 0 8px;font-weight:600}textarea,select,input{width:100%;border:1px solid #999;border-radius:0;padding:10px;background:#fff;color:inherit;font:15px system-ui}textarea{height:140px;resize:vertical}button{border:1px solid #222;border-radius:0;padding:10px 14px;margin:12px 7px 0 0;background:#222;color:#fff;font:15px system-ui;cursor:pointer}button.secondary{background:#fff;color:#222;border-color:#888}button:disabled{opacity:.5;cursor:wait}ol{padding-left:23px}li{padding:5px 0;font-size:14px;line-height:1.5}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto;background:#f5f5f5;padding:12px;border:1px solid #ddd}.notice{color:#444;font-size:14px}.toolbar{display:flex;gap:10px;flex-wrap:wrap}#decision{font-size:23px;font-weight:700;margin:12px 0}#status{font-size:14px}#summary{white-space:pre-wrap;font-size:15px}.compact{font-size:14px}.tools{display:flex;gap:8px;flex-wrap:wrap}.tools span{border:1px solid #ccc;padding:4px 7px;font:12px ui-monospace,monospace}@media(max-width:800px){.grid{grid-template-columns:1fr}h1{font-size:25px}main{padding:0 16px}}
</style><main>
<header><h1>短信分流与人工复核 Agent</h1><p class="muted">输入短信后运行分析，查看分类和链接检查结果，再确认处理意见。</p><span class="badge warn" id="connection">正在读取配置</span></header>
<div class="grid"><section class="panel"><h2>任务设置</h2><label for="mode">运行方式</label><select id="mode"><option value="llm">真实大模型 API · 需配置密钥</option><option value="demo">离线协议演示 · 模拟LLM响应</option><option value="local">传统工作流基线 · 不使用LLM</option></select><p class="notice" id="modeNote">真实模式将短信与工具结果发送至你配置的API服务。密钥仅在服务器端配置。</p><label for="goal">任务目标</label><input id="goal" maxlength="500" value="分析短信，检查相关证据，给出分流建议并等待我确认。"><label for="sms">待分析短信</label><textarea id="sms" maxlength="5000">Congratulations! You won a FREE prize. Claim your cash now at https://bit.ly/example</textarea><div><button id="go">运行 Agent</button><button class="secondary" id="ham">正常样例</button><button class="secondary" id="cn">中文边界</button></div><p class="muted compact" id="progress">不会访问短信中的链接，也不会发送或删除短信。</p></section>
<section class="panel"><h2>执行记录</h2><div class="tools"><span>classify_sms</span><span>inspect_links</span><span>lookup_policy</span><span>submit_assessment</span></div><p class="muted compact">API返回工具请求 → 本地校验并执行 → 回传工具结果 → 模型决定下一步。</p><ol id="steps"><li>等待开始任务。</li></ol><details><summary>查看每次工具返回的实际证据</summary><pre id="tools">尚无工具结果</pre></details></section></div>
<section class="panel" id="result" hidden><div class="toolbar"><span class="badge" id="status"></span><span class="badge warn" id="source"></span></div><div id="decision"></div><p id="score"></p><p id="summary"></p><p class="compact" id="evidence"></p><p class="compact" id="links"></p><div id="reviewPanel" hidden><h3>确认处理意见</h3><select id="reviewDecision"><option value="likely_ham">确认为倾向正常</option><option value="likely_spam">确认为疑似垃圾短信</option><option value="review">保持人工复核</option></select><button id="confirm">确认并完成任务</button><small>确认只更新本次任务，不自动重新训练模型。</small></div><button class="secondary" id="export">导出任务报告 JSON</button><details><summary>完整结构化结果</summary><pre id="json"></pre></details></section>
<section class="panel compact"><strong>配置方法</strong><p>复制项目根目录的 .env.example 为 .env，填写 LLM_API_KEY、LLM_BASE_URL 和支持工具调用的 LLM_MODEL，然后重启服务。交付文件中的 API Key 留空。</p><p class="muted">离线协议演示用于核验程序与工具，不代表真实大模型能力。机器学习分类实验与大模型任务评估分别统计。</p></section></main>
<script>
const $=x=>document.getElementById(x);let current=null;
const notes={llm:'真实模式将短信与工具结果发送至你配置的API服务。密钥仅在服务器端配置。',demo:'当前为离线协议演示：工具由真实代码执行，模型响应来自模拟脚本；没有调用大模型。',local:'当前为传统分类与规则基线：用于对照，不包含大模型规划。'};
$('mode').onchange=()=>{$('modeNote').textContent=notes[$('mode').value]};
$('ham').onclick=()=>{$('sms').value='Hi, I will meet you at the library after class.'};$('cn').onclick=()=>{$('sms').value='您的快递异常，请点击链接。'};
function render(d){current=d;$('progress').textContent='执行结束：'+(d.status||'baseline');$('result').hidden=false;$('decision').textContent=d.label||d.error;$('status').textContent=d.status||'baseline';$('source').textContent=d.mode==='demo'?'模拟LLM · 非真实API':d.mode==='llm'?'真实API模式':'离线工作流基线';$('score').textContent=d.score==null?'未输出分类分数':'课程模型 spam 分数：'+d.score+'（不是诈骗概率）';$('summary').textContent=d.summary||'传统工作流给出的分流建议。';$('steps').replaceChildren(...(d.trace||[]).map(x=>{let li=document.createElement('li');li.textContent=x.state+'：'+x.detail;return li}));$('tools').textContent=JSON.stringify(d.tool_results||[],null,2);$('evidence').textContent='特征依据：'+((d.evidence||[]).map(e=>e.feature+' ('+e.contribution+')').join('，')||'无');$('links').textContent='链接结果：'+JSON.stringify(d.links||[]);$('json').textContent=JSON.stringify(d,null,2);$('reviewPanel').hidden=d.status!=='pending_review';if(d.status==='pending_review')$('reviewDecision').value=d.action;if(d.review?.state==='confirmed')$('summary').textContent+='\n人工最终决定：'+d.review.decision+'；任务已完成。'}
$('go').onclick=async()=>{$('go').disabled=true;$('progress').textContent='正在执行，请等待工具调用与模型响应……';$('result').hidden=true;try{let r=await fetch('/api/triage',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('sms').value,goal:$('goal').value,mode:$('mode').value})});let d=await r.json();render(d);$('progress').textContent='执行结束：'+(d.status||'baseline')}catch(e){$('progress').textContent='本地服务请求失败，请检查终端。'}finally{$('go').disabled=false}};
$('confirm').onclick=async()=>{if(!current)return;$('confirm').disabled=true;try{let r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:current.id,decision:$('reviewDecision').value})});render(await r.json())}finally{$('confirm').disabled=false}};
$('export').onclick=()=>{if(!current)return;let u=URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=u;a.download='sms-task-'+(current.id||'baseline')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
fetch('/api/status').then(r=>r.json()).then(d=>{$('connection').textContent=d.configured?'API配置已填写 · 连接尚需实际验证':'API 未配置，可选择离线演示'}).catch(()=>{$('connection').textContent='配置读取失败'});
</script></html>'''
STORE = ReviewStore()
class Handler(BaseHTTPRequestHandler):
    agent = Agent()
    def log_message(self, *a): pass
    def send_json(self, d, code=200):
        data = json.dumps(d, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == '/api/status':
            s = Settings.from_env()
            return self.send_json({'configured': not s.missing(), 'missing': s.missing(), 'tools': [t['function']['name'] for t in TOOLS],
                                   'live_api_verified': False, 'modes': ['llm', 'demo', 'local']})
        if self.path != '/': return self.send_json({'error': 'not found'}, 404)
        self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(HTML.encode())
    def do_POST(self):
        if self.path not in ('/api/triage', '/api/review'): return self.send_json({'error': 'not found'}, 404)
        if self.headers.get('Origin') not in (None, f'http://{self.headers.get("Host")}'):
            return self.send_json({'error': 'origin rejected'}, 403)
        try:
            n = int(self.headers.get('Content-Length', '0'))
            if n <= 0 or n > 30000: raise ValueError('body size')
            data = json.loads(self.rfile.read(n))
            if not isinstance(data, dict): raise ValueError('body type')
            if self.path == '/api/review':
                try: return self.send_json(STORE.confirm(data.get('id'), data.get('decision')))
                except KeyError: return self.send_json({'error': '任务不存在或已过期'}, 404)
                except ValueError: return self.send_json({'error': '任务不可确认或决定无效'}, 409)
            mode = data.get('mode', 'llm')
            if mode not in ('llm', 'demo', 'local'): raise ValueError('mode')
            if mode == 'local': result = dict(self.agent.run(data.get('text')), mode='local', status='baseline')
            else:
                result = LLMAgent(mode=mode, baseline=self.agent).run(data.get('text'), data.get('goal', '分析短信，形成建议并等待人工确认。'))
                if result['status'] == 'pending_review': STORE.add(result)
            self.send_json(result)
        except (ValueError, TypeError, AttributeError): self.send_json({'error': 'invalid JSON request'}, 400)
        except Exception: self.send_json({'error': 'service unavailable; 请人工复核'}, 503)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--text'); parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--mode', choices=['llm', 'demo', 'local'], default='llm'); args = parser.parse_args()
    if args.text is not None:
        result = dict(Agent().run(args.text), mode='local', status='baseline') if args.mode == 'local' else LLMAgent(mode=args.mode).run(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f'本地界面 http://127.0.0.1:{args.port}', flush=True)
        ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()

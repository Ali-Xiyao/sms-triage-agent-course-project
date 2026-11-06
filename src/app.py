"""Local UI + JSON API + CLI. Standard-library HTTP server, binds loopback only."""
import sys,json,argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from core import Agent,ROOT
HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>短信风险分流 Agent</title><style>
body{font:17px system-ui;background:#f5f6fa;color:#202b3c;margin:40px auto;max-width:1000px;padding:0 25px}h1{font-size:32px}small{color:#53657d}textarea{box-sizing:border-box;width:100%;height:140px;padding:16px;font:17px system-ui;border:1px solid #aebbd0;border-radius:10px}button{padding:12px 20px;margin:14px 8px 14px 0;border:0;border-radius:7px;background:#234e77;color:white;font-size:16px;cursor:pointer}section{background:white;border-radius:12px;padding:24px;margin-top:20px}li{padding:7px}pre{white-space:pre-wrap;word-break:break-all;font-size:14px}.tag{color:#234e77}#decision{font-size:24px}details{margin-top:18px}</style>
<h1>短信风险分流 Agent</h1><p>英文短信本地分析 · 机器学习分类 · 静态链接检查 · 人工复核</p><small>不访问链接，不发送消息，不删除短信。仅支持英文为主的文本；模型分数不等于诈骗概率。</small>
<section><label for="sms">短信内容</label><textarea id="sms" maxlength="5000">Congratulations! You won a FREE prize. Claim your cash now at https://bit.ly/example</textarea><button id="go">运行 Agent</button><button id="ham">填入正常样例</button><button id="cn">填入中文边界样例</button></section>
<section id="result" hidden><div class="tag">分流结果</div><h2 id="decision"></h2><p id="score"></p><h3>执行轨迹</h3><ol id="steps"></ol><h3>特征依据</h3><p id="evidence"></p><h3>链接检查</h3><p id="links"></p><details><summary>查看结构化结果</summary><pre id="json"></pre></details></section>
<script>const $=x=>document.getElementById(x);$('ham').onclick=()=>{$('sms').value='Hi, I will meet you at the library after class.'};$('cn').onclick=()=>{$('sms').value='您的快递异常，请点击链接。'};$('go').onclick=async()=>{ $('go').disabled=true;try{let r=await fetch('/api/triage',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('sms').value})});let d=await r.json();$('result').hidden=false;$('decision').textContent=d.label||d.error;$('score').textContent=d.score==null?'未输出模型分数':'Spam 模型分数 '+d.score;$('steps').replaceChildren(...(d.trace||[]).map(s=>{let li=document.createElement('li');li.textContent=s.state+'：'+s.detail;return li}));$('evidence').textContent=(d.evidence||[]).map(e=>e.feature+' ('+e.contribution+')').join('，')||'无可用特征';$('links').textContent=JSON.stringify(d.links||[]);$('json').textContent=JSON.stringify(d,null,2)}catch(e){alert('本地服务请求失败，请检查终端')}finally{$('go').disabled=false}};</script></html>'''
class Handler(BaseHTTPRequestHandler):
    agent=Agent()
    def log_message(self,*a): pass  # Do not log private input.
    def send_json(self,d,code=200):
        data=json.dumps(d,ensure_ascii=False).encode();self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path!='/':return self.send_json({'error':'not found'},404)
        self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers();self.wfile.write(HTML.encode())
    def do_POST(self):
        if self.path!='/api/triage':return self.send_json({'error':'not found'},404)
        if self.headers.get('Origin') not in (None, f'http://{self.headers.get("Host")}'):return self.send_json({'error':'origin rejected'},403)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<=0 or n>30000:raise ValueError('invalid body size')
            data=json.loads(self.rfile.read(n)); result=self.agent.run(data.get('text'));self.send_json(result)
        except (ValueError,TypeError,AttributeError):self.send_json({'error':'invalid JSON request'},400)
        except Exception:self.send_json({'error':'service unavailable; 请人工复核'},503)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--text');parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    if args.text is not None:print(json.dumps(Agent().run(args.text),ensure_ascii=False,indent=2))
    else:print(f'本地界面 http://127.0.0.1:{args.port}',flush=True);ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

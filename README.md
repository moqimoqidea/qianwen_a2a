# 千问 A2A 实时天气 MVP

这是一个面向千问开放平台联调的 Python MVP。服务遵循 A2A 0.3.0，使用官方
`a2a-sdk` 0.3 系列，将 A2A 的非流式/流式请求分别转发到现有千问
Responses 兼容模型接口，并保留联网搜索能力。

## 已实现

- `GET /.well-known/agent-card.json`：A2A Agent Card，声明 JSON-RPC 和流式能力。
- `POST /a2a`：支持 `message/send` 与 `message/stream`。
- 流式帧按千问规范输出：`task(submitted)` → `artifact-update` 增量 →
  `status-update(completed)`。
- 模型推理摘要映射为 `artifact.name=reasoning`，正文映射为
  `artifact.name=message`。
- `POST /agent/event/callback`：接收 `audit.failed` 与
  `conversation.terminated`，并中止匹配的活动请求。
- 可选千问请求签名校验。
- 模型密钥只从环境变量读取。

## 本地启动

需要 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

编辑 `.env`，至少填写：

```dotenv
QWEN_API_URL=https://你的模型地址/api/v2/apps/protocols/compatible-mode/v1/responses
QWEN_API_KEY=你的新密钥
A2A_PUBLIC_URL=https://千问平台可访问的公网域名
```

启动：

```bash
qwen-a2a
```

### 后台运行测试服务

项目提供了后台启动和停止脚本。脚本会自动切换到项目根目录，从 `.env` 读取配置，
将输出追加到 `logs/qwen-a2a.log`，并把进程号写入 `logs/qwen-a2a.pid`：

```bash
./scripts/start-server.sh
tail -f logs/qwen-a2a.log
./scripts/stop-server.sh
```

等价的手动 `nohup` 命令如下：

```bash
mkdir -p logs
nohup .venv/bin/qwen-a2a >> logs/qwen-a2a.log 2>&1 &
echo $! > logs/qwen-a2a.pid
```

手动停止服务：

```bash
kill "$(cat logs/qwen-a2a.pid)"
rm -f logs/qwen-a2a.pid
```

健康检查：

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/.well-known/agent-card.json
```

## 协议联调

非流式请求会令上游模型使用 `stream: false`：

```bash
curl http://127.0.0.1:8000/a2a \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "answer-001",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "question-001",
        "contextId": "session-001",
        "role": "user",
        "parts": [{"kind": "text", "text": "今天北京天气怎么样？"}]
      }
    }
  }'
```

把 `method` 改为 `message/stream` 后，上游模型使用 `stream: true`，服务返回
`text/event-stream`。

在千问开放平台配置：

- Agent Card：`https://你的域名/.well-known/agent-card.json`
- A2A 服务地址：`https://你的域名/a2a`
- 事件回调：`https://你的域名/agent/event/callback`

若平台侧启用接口签名，同时设置 `QWEN_CLIENT_ID` 与
`QWEN_CLIENT_SECRET`。服务按文档校验
`SHA-256(Method&Path&TM&ClientId&ClientSecret&Nonce)`，默认允许 5 分钟时钟偏差。

## 验证

```bash
ruff check .
pytest
```

## MVP 边界

任务和取消状态目前保存在单进程内存中。联调阶段请使用单 worker；正式部署前应将
任务存储和取消协调替换为 Redis/数据库，并配置 HTTPS、限流、监控和密钥轮换。

原示例脚本中曾出现明文模型密钥，已从代码移除。该密钥应立即在控制台轮换，之后只
通过 `.env` 或部署平台的密钥管理服务注入。

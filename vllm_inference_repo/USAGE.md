# USAGE.md — 环境搭建与脚本使用指南

## 一、环境准备

### 1.1 WSL2 + Ubuntu 22.04

```powershell
# Windows 管理员 PowerShell
wsl --install -d Ubuntu
wsl --set-version Ubuntu 2
```

### 1.2 Ubuntu 内依赖安装

```bash
sudo sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
sudo apt update
sudo apt install -y python3-pip python3-venv build-essential git curl wget

python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate

mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

cd /path/to/vllm_inference_repo
pip install -r requirements.txt
```

### 1.3 关键版本兼容性

| 组件 | 版本 | 原因 |
|------|------|------|
| vLLM | **0.9.2** | 0.20+ 要 CUDA 13 + 驱动 580+，主流显卡不兼容 |
| torch | **2.7.0** | 与 vLLM 0.9.2 匹配 |
| transformers | **4.52.4** | transformers 5.x 与 vLLM 0.9.2 的 aimv2 注册冲突 |

验证：
```bash
python -c "import vllm, torch; print('vLLM:', vllm.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# 预期：vLLM: 0.9.2 / CUDA: True / GPU: NVIDIA GeForce RTX 2070
```

## 二、启动 vLLM Server

所有 `demo_*.py` 都通过 OpenAI 兼容 API 调用 server，必须先启动。

### 2.1 启动

```bash
cd src/
bash start_server.sh
# 等待 ~20 秒，看到 "Application startup complete"
```

### 2.2 验证

新开终端：
```bash
# 列出已加载模型
curl http://localhost:8000/v1/models

# 简单对话
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2-0.5b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'
```

### 2.3 停止

```bash
fuser -k 8000/tcp
```

## 三、各脚本使用

所有脚本位于 `src/`，运行前确保 venv 已激活且 server 已启动（`bench_throughput.py` 除外，它需要先停 server）。

### 3.1 bench_throughput.py — 吞吐对比

```bash
# 先停 vLLM server 释放显存
fuser -k 8000/tcp

python bench_throughput.py
```

**三路对比**：transformers 串行（一次一条）/ transformers batch=8（手动 padding）/ vLLM 批处理（continuous batching）

**产出**：
- 终端表格：总耗时 / QPS / tokens/s
- `outputs/throughput_results.json`：原始数据
- `outputs/throughput_comparison.png`：三路对比柱状图

跑完后重启 server：
```bash
bash start_server.sh
```

### 3.2 demo_function_call.py — 核心约束解码对比

```bash
# 跑两个工具共 100 个用例
python demo_function_call.py

# 只跑一个
python demo_function_call.py --tool stock
python demo_function_call.py --tool order
```

**两个工具**：
- `get_stock_quote`：金融股价查询，schema 含 string + enum + regex + array + minItems
- `create_order`：电商下单，schema 含 integer 范围 + 手机号正则 + 多枚举

**每个工具 50 条**，三种模式对比（裸 prompt / response_format / guided_json）。产出终端表格 + 典型失败案例 + `outputs/function_call_results.json`。

### 3.3 demo_guided_choice.py — 枚举约束

```bash
python demo_guided_choice.py
```

金融意图路由场景（查股价 / 查财报 / 查新闻 / 对比分析 / 其他）。对比裸 prompt 和 guided_choice 的输出合法率与分类准确率。

### 3.4 demo_guided_regex.py — 正则约束

```bash
python demo_guided_regex.py
```

两个任务：日期标准化（→ YYYY-MM-DD）和股票代码抽取（→ 6 位数字）。

### 3.5 demo_guided_json.py — JSON Schema 基础

```bash
python demo_guided_json.py
```

财报问答意图三元组抽取（公司 / 年度 / 指标）。对比三种模式在 year 范围约束、metric 枚举上的差异。

### 3.6 demo_response_format.py — OpenAI 标准方式

```bash
python demo_response_format.py
```

新闻情感分类 + 置信度 + 关键词。展示 `response_format={"type": "json_object"}` 与裸 prompt 的 JSON 合法率差异。

## 四、作为模块调用

### 4.1 通过 OpenAI 客户端调用本地 server

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# guided_json 约束解码示例
resp = client.chat.completions.create(
    model="qwen2-0.5b",
    messages=[{"role": "user", "content": "查茅台股价"}],
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "pattern": r"^\d{6}$"},
                "market": {"type": "string", "enum": ["SH", "SZ", "BJ"]},
                "price": {"type": "number"},
            },
            "required": ["symbol", "market"],
        }
    },
    temperature=0,
)
print(resp.choices[0].message.content)
```

### 4.2 离线直接调用 vLLM engine（不走 HTTP）

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen2-0.5B-Instruct",
    max_model_len=2048,
    gpu_memory_utilization=0.6,
    dtype="float16",
    enforce_eager=True,
)

sampling = SamplingParams(
    temperature=0, max_tokens=100,
    guided_decoding=GuidedDecodingParams(json=schema),
)
outputs = llm.generate(["prompt 1", "prompt 2"], sampling)
```

## 五、常见问题

| 问题 | 解法 |
|------|------|
| `ModuleNotFoundError: No module named 'vllm'` | `source ~/vllm_env/bin/activate` |
| `torch.cuda.is_available()` False | 降级到 `vllm==0.9.2` + `transformers==4.52.4` |
| `aimv2 is already used by a Transformers config` | `pip install transformers==4.52.4` |
| `Connection refused` | vLLM server 没启动，`bash start_server.sh` |
| bench 跑起来显存溢出 | 必须先停 vLLM server：`fuser -k 8000/tcp` |
| xgrammar fallback 警告 | 正常，硬件限制。outlines 可用但高级约束覆盖不全 |
| FlashAttention-2 不可用 | RTX 2070 Volta 架构不支持，自动 fallback 到 XFormers，性能相近 |

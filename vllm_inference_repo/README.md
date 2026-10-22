# vLLM Inference & Constrained Decoding

> 基于 vLLM 的 LLM 推理服务部署 + 约束解码工程实践：OpenAI 兼容 Server、吞吐对比、guided_choice / guided_regex / guided_json / response_format 四种解码约束在 Function Call 场景的可靠性量化。

---

## 核心成果

### 吞吐加速（Qwen2-0.5B-Instruct / RTX 2070 8GB / WSL2）

| 模式 | 50 请求总耗时 | QPS | Generation tok/s | 相对 vLLM |
|------|--------------|-----|------------------|-----------|
| transformers 串行 | 100.03s | 0.50 | 36 | 0.02× |
| transformers batch=8（padding）| 20.62s | 2.43 | 174 | 0.09× |
| **vLLM continuous batching** | **1.90s** | **26.26** | **1755** | **1.00×** |

**vLLM 相对 transformers 串行加速 52.5×；相对 batch=8 加速 10.8×。**

### 约束解码效果（Function Call，双工具 × 50 条 × 3 模式 = 300 次对比）

**get_stock_quote（金融股价查询，含 6 位正则 + enum + minItems）**

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 语法合法 | 86% | 100% | 100% |
| 完整 schema 通过 | **60%** | **68%** | **100%** |
| 平均延迟 | 0.62s | 0.83s | 3.06s |

**create_order（电商下单，含 integer 范围 + 手机号正则 + 多枚举）**

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 语法合法 | 96% | 100% | 100% |
| 完整 schema 通过 | **42%** | **42%** | **90%** |
| 平均延迟 | 0.76s | 1.02s | 4.29s |

关键观察：
- `response_format={"type":"json_object"}` 只保证 JSON 语法合法，不保证字段值语义正确（裸 prompt 42% vs response_format 还是 42%）
- `guided_json` 对 string/enum/regex/array 等约束 100% 生效，对 integer 范围（minimum/maximum）受 xgrammar/outlines fallback 能力限制（本项目 RTX 2070 上回落到 outlines，未强制整数边界）
- guided_json 延迟约为裸 prompt 的 3-5 倍（FSM 构建 + 每步 logit mask）

---

## 环境

| 组件 | 版本 |
|------|------|
| GPU | NVIDIA GeForce RTX 2070 8GB |
| Driver | 595.95 / CUDA 13.2 |
| OS | Windows 11 + WSL2 Ubuntu |
| vLLM | 0.9.2 |
| torch | 2.7.0 |
| transformers | 4.52.4 |
| Python | 3.10 |

为什么选择 vLLM 0.9.2：0.20+ 版本依赖 CUDA 13 runtime + 驱动 580+，RTX 2070/4060 等笔记本常见驱动版本（566.x / 595.x）都在 CUDA 12.x，0.9.2 + torch 2.7+cu118 是最稳妥的兼容组合。

---

## 快速开始

```bash
# 1. 创建 venv 并安装依赖
python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate
pip install -r requirements.txt

# 2. 启动 vLLM server（后台监听 0.0.0.0:8000）
cd src/
bash start_server.sh

# 3. 验证
curl http://localhost:8000/v1/models

# 4. 跑 benchmark（需先停 server 释放显存）
fuser -k 8000/tcp
python bench_throughput.py   # 产出 outputs/throughput_results.json + .png
bash start_server.sh

# 5. 跑约束解码 demo
python demo_function_call.py          # 核心：双工具 × 50 条 × 3 模式
python demo_guided_choice.py          # 枚举约束
python demo_guided_regex.py           # 正则约束
python demo_guided_json.py            # JSON Schema 基础
python demo_response_format.py        # OpenAI 标准 json_object
```

详细说明见 [ARCHITECTURE.md](ARCHITECTURE.md) 和 [USAGE.md](USAGE.md)。

---

## 目录结构

```
vllm_inference_repo/
├── README.md                   # 本文件
├── ARCHITECTURE.md             # 架构与技术选型
├── USAGE.md                    # 环境搭建 + 脚本使用
├── requirements.txt            # Python 依赖
├── src/
│   ├── start_server.sh         # 一键启动 vLLM OpenAI server
│   ├── bench_throughput.py     # 三路吞吐对比（transformers 串行 / batch / vLLM）
│   ├── demo_guided_choice.py   # 枚举约束解码
│   ├── demo_guided_regex.py    # 正则约束解码
│   ├── demo_guided_json.py     # JSON Schema 约束解码
│   ├── demo_response_format.py # OpenAI 标准 response_format
│   └── demo_function_call.py   # 核心：双工具 × 50 × 3 模式对比
└── outputs/
    ├── throughput_results.json         # 吞吐对比原始数据
    ├── throughput_comparison.png       # 三路对比柱状图
    └── function_call_results.json      # 约束解码详细结果
```

---

## 核心机制

**PagedAttention**：传统 KV cache 连续内存分配，batch 必须 pad 到最长 prompt，显存浪费严重。vLLM 按 16 token 为一个 block 动态分配，消除 padding 浪费，在固定显存上可容纳更大的 batch size。

**Continuous Batching**：不等 batch 内最长请求生成完毕，短请求一完成立刻插入新请求，GPU 利用率从串行时的 ~20% 拉到近 100%。

**Guided Decoding（FSM）**：vLLM 内部用 xgrammar 将枚举/正则/JSON Schema 编译成有限状态自动机（FSM），每步解码时根据当前状态屏蔽非法 token 的 logit（设为 `-inf`），仅从合法集采样。不修改模型权重，不改参数，仅增加每步一次向量 mask。

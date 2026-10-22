# ARCHITECTURE.md — vLLM 推理部署与约束解码

## 一、整体流水线

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │   Qwen2-0.5B-Instruct（0.94 GB on GPU, pretrain_models/）             │
 └──────────────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
 [A] transformers            [B] transformers           [C] vLLM engine
     serial                      batch=8                   (paged attn +
                                                          continuous batching)
        │                          │                          │
        │                          │                          │
        ▼                          ▼                          ▼
   36 tok/s                   174 tok/s                  1755 tok/s
    0.50 QPS                    2.43 QPS                   26.26 QPS
                                                             │
                                                             ▼
                                                  启动 OpenAI 兼容 HTTP 服务
                                                  (src/start_server.sh)
                                                             │
    ┌────────────────────────────────────────────────────────┴────────────────┐
    ▼                    ▼                    ▼                    ▼          ▼
demo_guided_      demo_guided_        demo_guided_       demo_response    demo_function
 choice.py         regex.py            json.py           _format.py        _call.py
（枚举）          （正则）            （Schema）          （JSON mode）      ★（综合）

                                                                       双工具 × 50×3 对比
                                                                       outputs/*.json
                                                                       throughput_*.png
```

## 二、技术选型

### 2.1 基座模型：Qwen2-0.5B-Instruct

**选型原因**：
- 预训练模型已就绪，无需重新下载
- 0.5B 足够小，RTX 2070 8GB 显存可完整跑三路对比
- 0.5B 的"能力边界"恰好能暴露约束解码的价值——它会真实犯错（字段拼错、枚举用自然语言、JSON 断尾），让 guided_json 提升从 42% → 100% 的效果被清晰展示
- Qwen2 tokenizer 中文支持好，适合金融领域中文 prompt

### 2.2 推理引擎：vLLM 0.9.2

**为什么不选最新版**：vLLM 0.20+ 依赖 torch 2.11 + CUDA 13 runtime，要求 NVIDIA 驱动 580+。RTX 2070/4060 等笔记本常见驱动（566.x / 595.x，CUDA 12.x）不兼容。0.9.2 + torch 2.7+cu118 是覆盖现役显卡的最稳妥组合。

**关键启动参数**（`start_server.sh`）：

| 参数 | 值 | 理由 |
|------|-----|------|
| `--max-model-len` | 2048 | 0.5B 模型不需要长上下文，省 KV cache |
| `--gpu-memory-utilization` | 0.6 | 留 40% 显存给其他程序（bench 需同时跑 transformers）|
| `--dtype` | float16 | RTX 2070 Volta 架构不支持 bf16 + FlashAttention-2 |
| `--enforce-eager` | 加 | 关 CUDA graph 首次启动快 5~10 秒 |
| `--host 0.0.0.0` | | 允许 Windows 侧浏览器/PowerShell 访问 |

### 2.3 约束解码：guided_choice / guided_regex / guided_json

vLLM 内部用 **xgrammar + outlines** 实现约束解码：
1. 解码前把 Schema/正则/枚举构建成 FSM（有限状态自动机）
2. 每步解码时，根据当前 FSM 状态把非法 token 的 logit 设为 `-inf`
3. Softmax 后只能从合法 token 集采样

**重要限制**：本项目 GPU（RTX 2070 Volta，Compute Capability 7.5）触发了 xgrammar fallback 到 outlines 的路径（见 server 日志 `xgrammar does not support advanced JSON schema features like string length, item limits, or property bounds. Falling back to use outlines instead`）。outlines 对 `minimum/maximum` 等 integer range 约束支持不完整，这是 create_order 的 guided_json 没拿到 100% 而只有 90% 的直接原因（5 个失败都是数量超出 [1, 100] 范围）。

### 2.4 评估框架：jsonschema + 分层指标

三层递进指标让三种模式的差异一目了然：
- **JSON 语法合法**：能被 json.loads 解析
- **必选字段齐全**：required 数组里的字段都存在
- **完整 schema 通过**：jsonschema.validate 通过，含类型/正则/枚举/范围等全部约束

## 三、实验结果

### 3.1 吞吐对比（Qwen2-0.5B / RTX 2070 8GB / WSL2）

| 模式 | 50 请求总耗时 | QPS | Generation tok/s | 相对 vLLM |
|------|--------------|-----|------------------|-----------|
| [A] transformers 串行 | **100.03s** | 0.50 | 36 | 0.02× |
| [B] transformers batch=8（padding）| 20.62s | 2.43 | 174 | 0.09× |
| [C] vLLM continuous batching | **1.90s** | **26.26** | **1755** | **1.00×** |

**vLLM 相对 transformers 串行加速 52.5×；相对 batch=8 加速 10.8×。**

结果解读：
- A → B 加速 4.9×：简单批处理能大幅提速，但 prompts 长短不一，padding 到最长有大量 token 浪费
- B → C 加速 10.8×：continuous batching + PagedAttention 才是质变：短请求完成后立即插入新请求，KV cache 按 block 动态分配，batch size 可动态到 20~40，GPU 利用率接近 100%

### 3.2 Function Call 约束解码效果

**get_stock_quote（金融股价查询，50 条测试）**

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 语法合法 | 86% | 100% | 100% |
| 必选字段齐全 | 86% | 100% | 100% |
| **完整 schema 通过** | **60%** | **68%** | **100%** |
| 平均延迟 | 0.62s | 0.83s | 3.06s |

典型失败：裸 prompt 在字段值填自然语言 `"最高价"` 而非 enum 值 `"high"`；空数组 `fields: []` 违反 minItems。

**create_order（电商下单，50 条测试）**

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 语法合法 | 96% | 100% | 100% |
| 必选字段齐全 | 96% | 100% | 100% |
| **完整 schema 通过** | **42%** | **42%** | **90%** |
| 平均延迟 | 0.76s | 1.02s | 4.29s |

典型失败：裸 prompt 在手机号加 `+` 前缀（违反正则），payment_method 用 `"credit card"` 而非 enum 值 `"card"`。guided_json 修复了这些问题，但 5 个失败用例都是 **quantity 超出 [1, 100] 范围**（如 "给我 200 个鼠标" → quantity=200），属于 integer range 约束 fallback 未覆盖的场景。

### 3.3 核心结论

1. `response_format={"type":"json_object"}` 把 JSON 语法合法率从 ~90% 拉到 100%，但**字段语义准确率没有任何改善**（裸和 rf 在 create_order 都是 42%）
2. `guided_json=schema` 是唯一能把 string/enum/regex/array 等约束拉到 100% 的方式
3. integer 范围等高级约束在 xgrammar fallback 场景下可能失效，生产环境需要注意
4. guided_json 延迟约为裸 prompt 的 3-5 倍，FSM 构建开销约 1-2s（首次），后续可缓存

## 四、关键工程决策与踩坑

| 问题 | 根因 | 解法 |
|------|------|------|
| `torch.cuda.is_available()` False | vLLM 0.20.x 依赖 torch + CUDA 13，需驱动 580+ | 降级 `pip install vllm==0.9.2` |
| `aimv2 is already used by a Transformers config` | transformers 5.x 内置 aimv2，与 vLLM 0.9.2 冲突 | `pip install transformers==4.52.4` |
| FlashAttention-2 不可用 | RTX 2070 Volta 架构（Compute Capability 7.5）不支持 | 自动 fallback 到 XFormers backend |
| 同时跑 bench 显存不足 | 两份 0.5B 模型 + KV cache | 先停 vLLM server 再跑 bench |
| WSL2 pin_memory=False | vLLM 自动检测 WSL 关闭 pin_memory | 跨文件系统拷贝慢约 5%，可忽略 |
| xgrammar fallback 到 outlines | RTX 2070 上 xgrammar 不支持 string length / property bounds | outlines 可用但高级约束覆盖不全 |

## 五、目录结构

```
vllm_inference_repo/
├── README.md                   # 项目概览
├── ARCHITECTURE.md             # 本文档：技术方案与实验结果
├── USAGE.md                    # 环境搭建 + 脚本使用
├── requirements.txt            # Python 依赖
│
├── src/
│   ├── start_server.sh         # 一键启动 vLLM OpenAI server
│   ├── bench_throughput.py     # transformers vs vLLM 三路吞吐对比
│   ├── demo_guided_choice.py   # 枚举约束解码
│   ├── demo_guided_regex.py    # 正则约束解码
│   ├── demo_guided_json.py     # JSON Schema 约束解码（基础）
│   ├── demo_response_format.py # OpenAI 标准 response_format
│   └── demo_function_call.py   # 核心：双工具 × 50 × 3 模式对比
│
└── outputs/
    ├── throughput_comparison.png
    ├── throughput_results.json
    └── function_call_results.json
```

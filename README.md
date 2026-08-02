# rag-ner

本地/内网中文 NER HTTP 服务，使用 ModelScope 的
`iic/nlp_raner_named-entity-recognition_chinese-base-generic`（RaNER）。支持 macOS
Apple Silicon CPU，以及 Debian 13 x86_64 上的 NVIDIA Tesla T4（CUDA 12.9）。

RaNER 是固定类型的通用中文命名实体识别模型，只返回：人名（`PER`）、机构名（`ORG`）、
地名（`LOC`）和地缘政治实体（`GPE`）。它不是通用 UIE 模型，不能通过请求临时增加
“药物”“产品”等标签。

后续文档字段抽取的范围、NER 加工程解析路线、OCR 输入约定和当前待实现事项，统一记录在
[文档字段抽取技术决策](docs/document-extraction-decisions.md)。

## 安装

需要 Python 3.13 与 [uv](https://docs.astral.sh/uv/)。普通依赖经中科大镜像安装；Debian
GPU 环境使用针对 CUDA 12.9 的 PyTorch wheel。模型从 ModelScope 下载并缓存在
`./models`。服务启动时只读取本地缓存，不访问网络；模型缺失或不完整会直接启动失败。

```bash
uv sync --extra ml --locked
test -f .env || cp .env.example .env
uv run --extra ml --locked python scripts/download_model.py
```

首次运行和日常部署都只执行 `uv sync --extra ml --locked`。仅在有意修改
`pyproject.toml` 中的依赖配置时，才执行 `uv lock` 刷新锁文件，并将 `pyproject.toml` 与
`uv.lock` 一起提交。

### macOS Apple Silicon

保持 `.env` 中 `DEVICE=cpu`，然后启动：

```bash
uv run --extra ml --locked python -m rag_ner.main
```

为保证 ModelScope 的 RaNER pipeline 稳定运行，macOS 固定使用 CPU；不尝试自动切换到
MPS。

### Debian 13 + Tesla T4（CUDA 12.9）

`torch==2.13.0+cu129` 已在锁定依赖中。PyTorch wheel 自带匹配的 CUDA 运行时；不需要
替换机器的 CUDA Toolkit，但 NVIDIA 驱动必须支持 CUDA 12.9。

```bash
uv run --extra ml --locked python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 将 .env 的 DEVICE 改为 cuda:0 后启动
uv run --extra ml --locked python -m rag_ner.main
```

GPU 不可用、设备号不存在，或 macOS 配置了 GPU 时，服务会启动失败，不会静默回退 CPU。

## 调用

```bash
curl -s http://localhost:8002/v1/ner \
  -H 'Content-Type: application/json' \
  -d '{"text":"孙燕姿来自新加坡，目前在北京的腾讯音乐工作。"}'
```

响应：

```json
{
  "entities": [
    {"type": "PER", "start": 0, "end": 3, "span": "孙燕姿"},
    {"type": "GPE", "start": 5, "end": 8, "span": "新加坡"},
    {"type": "GPE", "start": 12, "end": 14, "span": "北京"},
    {"type": "ORG", "start": 15, "end": 19, "span": "腾讯音乐"}
  ]
}
```

请求文本默认最多 480 个字符，避免模型 512-token 上限导致静默截断；可用 `.env` 中的
`MAX_INPUT_CHARACTERS` 调低，最高允许 510。设置 `API_KEY` 后，请求需带
`Authorization: Bearer <API_KEY>`。

| 状态码 | 场景 | 说明 |
| --- | --- | --- |
| 401 | API key 缺失或错误 | 仅在配置 `API_KEY` 时出现。 |
| 422 | 文本为空、只有空白或超过字符上限 | 调整输入或 `MAX_INPUT_CHARACTERS`。 |
| 503 | NER 请求并发已满 | 响应带 `Retry-After: 1`，调用方应稍后重试。 |

可用 `GET /healthz` 检查进程存活，`GET /readyz` 检查模型已经加载完成。
模型加载包含一次真实预热推理，因此 `ready` 表示模型已完成设备初始化。生产环境应在
Nginx/Caddy 配置请求正文大小限制，并避免将 Uvicorn 直接暴露到公网。

## 测试

测试不下载模型，也不需要 CUDA：

```bash
uv sync --extra dev --locked
uv run --extra dev --locked pytest
uv run --extra dev --locked ruff check src scripts tests
```

当前 `tests/test_school_notice_extraction.py` 是已提交的待实现规格，完整测试暂时有一个已知
失败；原因和后续处理见[文档字段抽取技术决策](docs/document-extraction-decisions.md)。

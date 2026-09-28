# MetaSkill Lab

一个独立的 Builder → Harness → Target → 公开反馈 → Skill → Refine 核心。Builder 为任务分布构建可复用环境，Target 在具体任务中使用环境；Builder 从训练执行中总结 `when / provide / use` 支持技能，再修订环境。构建和修订只做接口修复，不按 Target 分数挑选候选。

## 内容

- **七种组件**：`instruction`、`memory`、`tools`、`context`、`controller`、`verification`、`workspace`。关闭的组件写 `null`；打开的组件提供受限 Python 函数源码。`tools` 可组合 adapter 提供的原生工具。
- **固定边界**：JSON schema、受限源码解释器、独立任务会话、工具与 token/步数/时间预算，以及 Target 的工具循环。生成的代码不经 Python `exec` / `eval` 执行。
- **两个角色**：Builder 只读公开分布说明、工具定义、已有 bundle/skill 和公开训练反馈；Target 执行 adapter 提供的任务与工具。原生评分或其他私有状态由 adapter 自己管理。
- **模型**：直接调用 [OpenAI Responses API](https://developers.openai.com/api/docs/guides/function-calling) 和 [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create)。`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` 从环境变量读取。没有额外 Python 依赖。

这是一套方法内核和 adapter 接口。要运行新的 benchmark，实现 `public_tools()`、`load_task(task_id)`；可选实现 `public_feedback(result)`。`examples/catalog_adapter.py` 展示最小接法。Builder 输入的公开分布说明不包含具体测试题。

## 快速运行

在仓库目录中：

```sh
python -m pip install -e .
export OPENAI_API_KEY="..."   # 或 ANTHROPIC_API_KEY

python -m meta_skill_ai4ai build \
  --adapter examples.catalog_adapter --provider openai --model gpt-4.1-mini \
  --card examples/benchmark_card.json --output /tmp/meta-bundle.json

python -m meta_skill_ai4ai run \
  --adapter examples.catalog_adapter --provider openai --model gpt-4.1-mini \
  --bundle /tmp/meta-bundle.json --task-id catalog-1 \
  --output /tmp/meta-episode.json

python -m meta_skill_ai4ai learn \
  --adapter examples.catalog_adapter --provider openai --model gpt-4.1-mini \
  --bundle /tmp/meta-bundle.json --episode /tmp/meta-episode.json \
  --output /tmp/meta-skills.json

python -m meta_skill_ai4ai refine \
  --adapter examples.catalog_adapter --provider openai --model gpt-4.1-mini \
  --card examples/benchmark_card.json --bundle /tmp/meta-bundle.json \
  --skills /tmp/meta-skills.json --output /tmp/meta-refined.json
```

Claude 用 `--provider anthropic --model <你的 Claude 模型 ID>` 和 `ANTHROPIC_API_KEY`。Builder 与 Target 可分别指定提供方和型号。每个任务创建新会话；继续同一个任务时可在 Python API 里显式传入 `ModularSession`。模型调用、原生工具调用、token 与耗时均受 `TargetBudget` 控制。

## Adapter 合约

```python
from meta_skill_ai4ai import TargetTask, Tool, ToolResult

def public_tools() -> list[Tool]:
    # 返回任务分布共有的工具名称、描述、JSON 参数 schema 和 handler。
    ...

def load_task(task_id: str) -> tuple[TargetTask, list[Tool]]:
    # 这里才读取具体任务，工具名称须与 public_tools() 一致。
    ...

def public_feedback(result) -> dict:  # 可选
    # 仅放入允许 Builder 看到的训练反馈。
    ...
```

`Tool.handler(arguments)` 返回 `ToolResult(ok, observation, error_type)`。当原生环境需要立即终止时，可设置 `terminal_reason`；基础设施错误设置 `infrastructure_failure=True`。Harness 的工作区是独立临时空间；任务原生文件与评分只通过 adapter 的工具和反馈暴露。

## 验证

```sh
python -m unittest discover -s tests -v
```

`tests/test_workflow.py` 覆盖构建、Target 工具调用、skill 反思、refine 及两种官方接口的请求/响应形状。核心解释器和运行时沿用原仓库的 `modular-v2.0` 结构与执行边界。

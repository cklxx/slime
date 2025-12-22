# 在本地快速检查 CP 的切分效果

如果只想验证 context parallel（CP）在切分 logits、mask 和重组输出时的行为，不想拉起完整的分布式/模型环境，可以使用新的 `examples/cp_dry_run.py` 脚本完成一次“干跑”。

## 用法

```bash
# 默认 total_length=16（prompt+response），response_length=8，cp_size=2
python examples/cp_dry_run.py

# 根据需要调整长度或模拟的 CP size
python examples/cp_dry_run.py --total-length 24 --response-length 10 --cp-size 4

# 对比 CP=1 与 CP=2 的迭代耗时（默认步数 50，默认预热 10）
python examples/cp_dry_run.py --benchmark
```

脚本会：

- 使用 `unittest.mock` 替换 Megatron CP 相关接口，避免真实的通信和 GPU 依赖；
- 逐个 CP rank 打印 chunk 切分范围、对应的 logits/tokens 段以及本地 loss mask；
- 构造一段假 logits，展示 `all_gather_with_cp` 如何在无通信环境下完成填充与重组，方便观察 CP 对齐逻辑。
- 如果加上 `--benchmark`，会分别在 CP=1（基准）和 CP=2 下做循环迭代，打印每个场景的平均单卡耗时，方便确认开启 CP 后训练耗时不应有明显增加。

这样可以在纯 CPU 环境快速确认 CP 切分是否符合预期，再决定是否启动完整训练/推理流程。

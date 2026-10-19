# miyu-embed

[Miyu](https://github.com/miyu-agent/miyu) 的内置语义模型：一个小程序包 `embed`，在本机照意思找记忆、知识库，不用联网。装：

```sh
miyu pkg install embed
```

- 程序 `miyu-embed`：照模型清单 `package/embed/model.toml` 载入一份 ONNX 模型（ONNX Runtime 静态链接在它里面），标准输入一行一句话，标准输出一行一个向量。协议见核心仓库 `docs/blueprint/recall.md` 第四条。
- 模型：granite-embedding-97m-multilingual-r2 的 4 位量化（Apache-2.0），文件在本仓库的 Release `models-granite-embedding-97m-multilingual-r2`，`tools/quantize_q4.py`、`tools/bpe_files.py` 能逐字节重现。
- 发版：打 tag `v<版本>`，`release.yml` 照官方的打包模板一种系统打一个包文件，软件源每天收一次。
- 依赖核心的 `miyu-recall`（模型清单的类型），照 git 钉住核心的一个提交；升级核心就是改 `Cargo.toml` 里的 `rev`、刷 `Cargo.lock`、过 CI。

核心仓库的施工单 R-5、S-7 记着它怎么来的。

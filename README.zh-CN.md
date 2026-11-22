<p align="center">
  <a href="README.md">English</a> | 简体中文
</p>

# Miyu 内置语义模型（miyu-embed）

> <sub>本文档由 Anthropic 的 Claude Fable 5.2 生成。</sub>

这个仓库是 [Miyu](https://github.com/miyu-agent/miyu) 的内置语义模型，也就是一个带程序的小软件包 `embed`。它在本机按照意思查找记忆和知识库，不需要联网。安装时运行：

```sh
miyu pkg install embed
```

关于这个软件包，需要了解以下几点：

- 程序 `miyu-embed`：按照模型清单 `package/embed/model.toml` 载入一份 ONNX 模型（ONNX Runtime 静态链接在程序里），从标准输入每行读取一句话，向标准输出每行写出一个向量。协议见核心仓库的 `internal/blueprint/recall.md` 第四条。
- 模型：`granite-embedding-97m-multilingual-r2` 的 4 位量化版本（Apache-2.0），文件放在这个仓库的发行版 `models-granite-embedding-97m-multilingual-r2` 里，用 `tools/quantize_q4.py`、`tools/bpe_files.py` 可以逐字节重现。
- 发布新版本：打标签 `v<版本>`，`release.yml` 会按照官方的打包模板为每种系统打包一个软件包文件，软件源每天收录一次。
- 依赖：这个软件包依赖核心的 `miyu-recall`（模型清单的类型），通过 git 固定到核心的一个提交；升级核心的方法是修改 `Cargo.toml` 里的 `rev`、更新 `Cargo.lock`，并且通过 CI。

## 许可证

这个仓库自己编写的程序使用 AGPL-3.0-or-later，详细的见 `LICENSE`。不是这个仓库编写的内容使用它们原来的许可证：模型 `granite-embedding-97m-multilingual-r2` 使用 Apache-2.0（在发行包里叫 `LICENSE-model`、`NOTICE-model`）；静态链接进来的 ONNX Runtime 1.28.0 使用 MIT（原文和它的第三方声明在 `licenses/onnxruntime/`，在发行包里叫 `LICENSE-onnxruntime`、`ThirdPartyNotices-onnxruntime.txt`）；发行包里的 `LICENSE` 是这个仓库的 AGPL；Rust 依赖使用各自的 `license` 里写明的许可证。

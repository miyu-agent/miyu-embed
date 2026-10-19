# miyu-embed 的规矩

照核心仓库 `miyu-agent/miyu` 的 AGENTS.md：中文注释、不吞错误、每个文件不超过 500 行、改了先写会红的测试。

- 只有一个程序 `miyu-embed` 和包文件夹 `package/embed/`。模型清单的类型住在核心的 `miyu-recall`，两边照同一份读。
- 换模型就是换包：改 `package/embed/model.toml`、Release 里的文件、`tests/real.rs` 的对照，`Cargo.toml` 和 `package.toml` 的版本一起升。
- 真模型的测试 `#[ignore]`：`MIYU_EMBED_MODEL_DIR`、`MIYU_EMBED_BPE_DIR` 指到装好的包文件夹。

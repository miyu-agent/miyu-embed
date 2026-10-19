//! `miyu-embed` 的测试（施工 R-5 上，`docs/blueprint/recall.md` 第四条）：分词（R-5 六补的字节级 BPE；WordPiece 2026-10-19 随 bge 拿掉了）照 Hugging Face 的结果逐个比，清单的读法，
//! 小程序的协议（手造的小模型，CI 里跑），真模型的向量（`#[ignore]`，照 `MIYU_EMBED_MODEL_DIR` 找文件，验收时跑；包的清单照核心的读法）。

mod bpe;
mod manifest;
mod package;
mod protocol;
mod real;
mod support;

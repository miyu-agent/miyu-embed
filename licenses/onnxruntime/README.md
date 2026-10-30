# ONNX Runtime 的许可证

`miyu-embed` 静态链着 ONNX Runtime 1.28.0（`ort-sys 2.0.0-rc.13` 下载的那一份）。这里是它在
`microsoft/onnxruntime` 仓库 `v1.28.0` 标签下的 `LICENSE`（MIT）和 `ThirdPartyNotices.txt`（它带进来的
protobuf、abseil、Eigen 等的声明）原文。发版时 `release.yml` 把它们拷进包里，名字是
`LICENSE-onnxruntime`、`ThirdPartyNotices-onnxruntime.txt`。

升 `ort` 时 ONNX Runtime 的版本变了，照新版本的标签重新取这两份。

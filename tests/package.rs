//! 包的清单（施工 R-5 三补，S-7 下从核心挪过来）：照核心的读法是小程序包、程序是 `miyu-embed`，写了版本和四种系统。

use std::path::PathBuf;

use miyu_config::package::{PackageKind, Target, read};

#[test]
fn the_package_is_a_worker_for_four_targets() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("package/embed/package.toml");
    let embed = read(&std::fs::read_to_string(&path).expect("有清单")).expect("写法对");
    assert_eq!(embed.kind, PackageKind::Worker);
    assert_eq!(
        embed.worker.as_ref().map(|worker| worker.program.as_str()),
        Some("miyu-embed")
    );
    assert_eq!(
        embed.version.as_deref(),
        Some(env!("CARGO_PKG_VERSION")),
        "清单的版本跟着 Cargo.toml"
    );
    assert_eq!(embed.targets, Target::ALL.to_vec());
}

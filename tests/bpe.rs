//! 字节级 BPE（`recall.md` 第四条第 5 款，施工 R-5 六补）：照 Hugging Face `tokenizers` 0.23.3 训练的小 BPE（`fixtures/tiny-bpe/`，
//! `fixtures/make_bpe.py` 造、`tools/bpe_files.py` 转）上照它的结果逐个比（`fixtures/bpe-tokens.json`）；截断、字里的特殊词、
//! 词表和合并坏了另外测。真的 granite 词表（`#[ignore]`）照 `MIYU_EMBED_BPE_DIR` 找转出的两个文件，和
//! `fixtures/granite-tokens.json` 比（`MIYU_EMBED_BPE_CASES` 指到另一份同样格式的就比那一份）。

use std::path::Path;

use miyu_embed::bpe::{FilesError, Tokenizer};
use serde::Deserialize;

use crate::support::{fixtures, read};

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<i64>,
    /// 截到几个（不写的不截）：仓库外那一份是照 512 截的。
    #[serde(default)]
    limit: Option<usize>,
}

const START: &str = "<|startoftext|>";
const END: &str = "<|return|>";

/// 放得下这些句子的上限：不截断。
const ALL: usize = 1 << 20;

fn load(dir: &Path) -> Tokenizer {
    let text = |name: &str| {
        std::fs::read_to_string(dir.join(name))
            .unwrap_or_else(|error| panic!("读不了 {name}：{error}"))
    };
    Tokenizer::new(&text("vocab.txt"), &text("merges.txt"), START, END).expect("读得出")
}

fn tiny() -> Tokenizer {
    load(&fixtures().join("tiny-bpe"))
}

fn cases(name: &str) -> Vec<Case> {
    serde_json::from_str(&read(name)).expect("合写法")
}

fn matches_all(tokenizer: &Tokenizer, cases: &[Case]) {
    for case in cases {
        let limit = case.limit.unwrap_or(ALL);
        assert_eq!(
            tokenizer.encode(&case.text, limit),
            case.ids,
            "{:?}",
            case.text
        );
    }
}

#[test]
fn every_sentence_matches_the_reference() {
    let cases = cases("bpe-tokens.json");
    assert!(cases.len() > 60, "{}", cases.len());
    matches_all(&tiny(), &cases);
}

#[test]
fn a_cut_keeps_the_end_and_equals_the_head_of_the_whole() {
    let tokenizer = tiny();
    for case in cases("bpe-tokens.json") {
        let whole = &case.ids;
        for limit in 2..14 {
            let got = tokenizer.encode(&case.text, limit);
            let want: Vec<i64> = if whole.len() <= limit {
                whole.clone()
            } else {
                whole[..limit - 1]
                    .iter()
                    .chain(whole.last())
                    .copied()
                    .collect()
            };
            assert_eq!(got, want, "{:?} 截到 {limit}", case.text);
        }
    }
    let ids = tokenizer.encode("x", 1);
    assert_eq!(ids.len(), 2, "比 2 小的照 2 算：{ids:?}");
}

#[test]
fn a_special_word_in_the_text_is_plain_text() {
    let tokenizer = tiny();
    let ids = tokenizer.encode("a<|return|>b<|startoftext|>", ALL);
    let (start, end) = (ids[0], ids[ids.len() - 1]);
    assert_eq!(ids.iter().filter(|id| **id == start).count(), 1, "{ids:?}");
    assert_eq!(ids.iter().filter(|id| **id == end).count(), 1, "{ids:?}");
}

/// 小 BPE 的两份原文。
fn tiny_files() -> (String, String) {
    (read("tiny-bpe/vocab.txt"), read("tiny-bpe/merges.txt"))
}

fn refused(vocab: &str, merges: &str) -> FilesError {
    Tokenizer::new(vocab, merges, START, END).expect_err("要拒")
}

#[test]
fn bad_files_are_refused_and_said() {
    let (vocab, merges) = tiny_files();
    let first = vocab.lines().next().expect("有词").to_string();
    let wrong = [
        (
            refused(&format!("{vocab}{first}\n"), &merges),
            format!("vocab has {first} twice"),
        ),
        (
            refused(&vocab.replace(END, "<|gone|>"), &merges),
            format!("vocab has no {END}"),
        ),
        (
            refused(&vocab, &format!("{merges}nospace\n")),
            format!(
                "merges line {} is not two tokens",
                merges.lines().count() + 1
            ),
        ),
        (
            refused(&vocab, "Ġ zzzz\n"),
            "merges line 1: zzzz is not in vocab".to_string(),
        ),
        (
            refused(&vocab, "x y\n"),
            "merges line 1: xy is not in vocab".to_string(),
        ),
        (
            refused(
                &vocab,
                &format!(
                    "{}\n{}\n",
                    merges.lines().next().expect("有"),
                    merges.lines().next().expect("有")
                ),
            ),
            "merges line 2 repeats an earlier pair".to_string(),
        ),
    ];
    for (error, said) in wrong {
        assert_eq!(error.to_string(), said);
    }
    assert_eq!(
        refused(&vocab.replace(START, "<|gone|>"), &merges),
        FilesError::Missing(START.to_string())
    );
}

/// 词表里没有的字节丢掉，照 `tokenizers`（granite 的词表没有大部分控制字符）。
#[test]
fn a_byte_not_in_the_vocab_is_dropped() {
    let (vocab, merges) = tiny_files();
    // 字节 0x01 换成的字是 U+0101：去掉它那一行（换成一个用不到的字，别的编号不动），合并里没有它。
    assert!(vocab.contains("\n\u{101}\n") && !merges.contains('\u{101}'));
    let tokenizer = Tokenizer::new(
        &vocab.replacen("\n\u{101}\n", "\n\u{4e00}\n", 1),
        &merges,
        START,
        END,
    )
    .expect("缺字节的也读得出");
    assert_eq!(tokenizer.encode("a\u{1}", ALL), tokenizer.encode("a", ALL));
    assert_eq!(
        tokenizer.encode("\u{1}\u{1}", ALL),
        tokenizer.encode("", ALL)
    );
    assert_ne!(
        tiny().encode("a\u{1}", ALL),
        tiny().encode("a", ALL),
        "词表里有的不丢"
    );
}

#[test]
fn ids_are_line_numbers_and_windows_line_ends_are_read() {
    let (vocab, merges) = tiny_files();
    let crlf = Tokenizer::new(
        &vocab.replace('\n', "\r\n"),
        &merges.replace('\n', "\r\n"),
        START,
        END,
    )
    .expect("Windows 的换行也认");
    matches_all(&crlf, &cases("bpe-tokens.json"));
    let start = vocab.lines().position(|word| word == START).expect("有");
    assert_eq!(
        tiny().encode("", ALL),
        [start as i64, start as i64 + 2],
        "行号就是编号：start、endoftext、return"
    );
}

#[test]
#[ignore = "要真的 granite 词表：MIYU_EMBED_BPE_DIR（tools/bpe_files.py 转出的目录）"]
fn the_granite_vocab_agrees_with_the_reference() {
    let dir = std::env::var_os("MIYU_EMBED_BPE_DIR").expect("设了 MIYU_EMBED_BPE_DIR");
    let tokenizer = load(Path::new(&dir));
    let cases: Vec<Case> = match std::env::var_os("MIYU_EMBED_BPE_CASES") {
        Some(path) => {
            serde_json::from_str(&std::fs::read_to_string(path).expect("读得了")).expect("合写法")
        }
        None => cases("granite-tokens.json"),
    };
    matches_all(&tokenizer, &cases);
    println!("{} 句一字不差", cases.len());
}

/// 审计第四轮（2026-11-02）：`encode` 里说「回溯超了上限才会出错（这条正则只往前看一个字，碰不到）」，可一百万个连着的空白
/// （`\s+(?!\S)` 那一支）就超了 `fancy-regex` 的回溯栈；出错以后剩下的整段当一段合并，切法和 `tokenizers` 不一样，后面多长都
/// 照字节合一遍。
#[test]
fn the_split_never_gives_up_on_a_long_run_of_spaces() {
    let split = fancy_regex::Regex::new(miyu_embed::bpe::SPLIT).expect("合写法");
    let text = format!("{}x", " ".repeat(1 << 20));
    let failed = split.find_iter(&text).find_map(Result::err);
    assert!(failed.is_none(), "切段出错：{failed:?}");
}

/// 审计第四轮：截断只在段和段之间看（`ids.len() >= keep`），一段之内不看：一长串不带空白的字（几 MB 的字母、汉字）整段照字节
/// 合并，只留前 512 个，花的时间、内存照整段的长度长。照 `limit` 截的，花的应该和一句短话差不多。
#[test]
fn one_long_word_costs_about_as_much_as_the_limit() {
    let tokenizer = tiny();
    let text = "a".repeat(4 << 20);
    let started = std::time::Instant::now();
    let ids = tokenizer.encode(&text, 512);
    let took = started.elapsed();
    assert_eq!(ids.len(), 512);
    assert!(
        took < std::time::Duration::from_millis(100),
        "4 MB 一段截到 512 个花了 {took:?}"
    );
}

/// 审计第四轮 E-1 的修法拿掉了 `\s+(?!\S)` 一支（长空白会让 `fancy-regex` 的回溯栈溢出），改在 `encode` 里退一个字：拿原来
/// 带那一支的正则逐个比，空白、换行、字母、数字、符号、全角空白随机拼，切出来的一段段要一模一样。
#[test]
fn the_split_without_lookahead_cuts_like_the_reference_regex() {
    let reference =
        fancy_regex::Regex::new(&miyu_embed::bpe::SPLIT.replace(r"|\s+", r"|\s+(?!\S)|\s+"))
            .expect("合写法");
    assert!(reference.as_str().contains("(?!\\S)"));
    let tokenizer = tiny();
    let alphabet: Vec<char> = " \t\n\r a1.é'x\u{3000}\u{a0}字".chars().collect();
    let mut state = 12345_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize
    };
    for _ in 0..100_000 {
        let text: String = (0..next() % 24)
            .map(|_| alphabet[next() % alphabet.len()])
            .collect();
        let want: Vec<&str> = reference
            .find_iter(&text)
            .map(|found| found.expect("短句不出错").as_str())
            .collect();
        assert_eq!(tokenizer.pieces(&text), want, "{text:?}");
    }
}

/// 截短的字节数落在字中间时往前退到字边界，不 panic；整段汉字、`window` 以内的都照常。
#[test]
fn the_cut_lands_on_a_char_boundary() {
    let tokenizer = tiny();
    for text in [
        "字".repeat(100_000),
        "é".repeat(100_000),
        "😀".repeat(100_000),
    ] {
        assert_eq!(tokenizer.encode(&text, 8).len(), 8);
    }
}

/// 按字节截短不动前 `limit` 个词：几百 KB 的混合文字截到 8 个，和只给开头一小段的结果一样。
#[test]
fn the_byte_cut_keeps_the_first_tokens() {
    let tokenizer = tiny();
    let unit = "hello  world 你好, it's 42 \n\n  x";
    let long = unit.repeat(20_000);
    let short = unit.repeat(40);
    for limit in [2, 3, 8, 20] {
        assert_eq!(
            tokenizer.encode(&long, limit),
            tokenizer.encode(&short, limit)
        );
    }
}

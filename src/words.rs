//! 词表、合并的紧凑存法（EM-3）：granite 的词表十八万条、合并四十一万条，原来存成两张标准库的哈希表，键是一个个单独分配的
//! 小字符串，常驻约 23 MB。这里把所有的词拼进一整块字，表里只存编号（开放寻址、线性探查），比的时候回那一整块里看；合并照
//! 先后存成一个数组，表里同样只存下标。查的结果和原来的两张表一样，常驻约 10 MB。

use std::hash::Hasher;

use crate::bpe::FilesError;

/// 槽空着。编号和下标都不会到这么大：词表、合并的条数在载入时查过，放不下的直接拒。
const EMPTY: u32 = u32::MAX;

/// 查词表、合并用的哈希：FxHash 的做法（一次一个字长，乘一个常数）。载入时要算一百多万次，标准库的 SipHash 慢一倍多；
/// 键是模型自己的词表和切出来的一段字，碰撞攻击无从谈起。乘法把差别推到高位，所以下标取高位（[`Slots::start`]）。
#[derive(Debug, Clone, Copy, Default)]
struct Fx(u64);

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0; 8];
            word.copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
        for byte in chunks.remainder() {
            self.add(u64::from(*byte));
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

impl Fx {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

fn hash_word(word: &str) -> u64 {
    let mut fx = Fx::default();
    fx.write(word.as_bytes());
    // 再搅一次：最后几个字节刚加进去、还没乘过，不搅的话它们只影响低位。
    fx.add(word.len() as u64);
    fx.finish()
}

fn hash_pair(left: u32, right: u32) -> u64 {
    let mut fx = Fx::default();
    fx.add(u64::from(left) << 32 | u64::from(right));
    fx.add(0);
    fx.finish()
}

/// 开放寻址的一张表：槽里只存编号，键在别处，比的时候由调用的一方回去看。槽数是 2 的幂，至多八成满。
#[derive(Debug, Clone)]
struct Slots {
    slots: Box<[u32]>,
    /// 槽数是 2 的几次方。
    bits: u32,
}

impl Slots {
    /// 能放 `count` 条的一张空表。
    fn new(count: usize) -> Slots {
        let size = (count + count / 4 + 1).next_power_of_two().max(2);
        Slots {
            slots: vec![EMPTY; size].into_boxed_slice(),
            bits: size.trailing_zeros(),
        }
    }

    fn start(&self, hash: u64) -> usize {
        // 取高位：`bits` 至少是 1，移位不会到 64。
        usize::try_from(hash >> (64 - self.bits)).unwrap_or(0)
    }

    /// 找键的哈希是 `hash`、`same` 认的那一个编号。
    fn find(&self, hash: u64, same: impl Fn(u32) -> bool) -> Option<u32> {
        let mask = self.slots.len() - 1;
        let mut at = self.start(hash);
        loop {
            let id = self.slots[at];
            if id == EMPTY {
                return None;
            }
            if same(id) {
                return Some(id);
            }
            at = (at + 1) & mask;
        }
    }

    /// 放进 `id`；`same` 认的已经在表里的，不放、交回 `false`。
    fn insert(&mut self, hash: u64, id: u32, same: impl Fn(u32) -> bool) -> bool {
        let mask = self.slots.len() - 1;
        let mut at = self.start(hash);
        loop {
            let there = self.slots[at];
            if there == EMPTY {
                self.slots[at] = id;
                return true;
            }
            if same(there) {
                return false;
            }
            at = (at + 1) & mask;
        }
    }
}

/// 词表：第几行（从 0 数）就是编号。
#[derive(Debug, Clone)]
pub(crate) struct Vocab {
    /// 所有的词首尾相接。
    text: Box<str>,
    /// 第几个词在 `text` 里到哪个字节为止（开头是前一个的结尾）。
    ends: Box<[u32]>,
    slots: Slots,
}

impl Vocab {
    /// 照一行一个词造。同一个词写了两行的拒；词表大到编号、字节数放不进 `u32` 的也拒（granite 那份十八万条、不到 2 MB，
    /// 离得很远）。
    pub(crate) fn new(lines: &str) -> Result<Vocab, FilesError> {
        let count = lines.lines().count();
        if u32::try_from(lines.len()).is_err() || u32::try_from(count).map_or(true, |n| n == EMPTY)
        {
            return Err(FilesError::TooMany);
        }
        let mut text = String::with_capacity(lines.len());
        let mut ends = Vec::with_capacity(count);
        let mut slots = Slots::new(count);
        for (id, word) in (0_u32..).zip(lines.lines()) {
            text.push_str(word);
            // 不超过 `lines.len()`，上面查过放得进。
            ends.push(u32::try_from(text.len()).unwrap_or(u32::MAX));
            let fresh = slots.insert(hash_word(word), id, |there| {
                word_at(&text, &ends, there) == word
            });
            if !fresh {
                return Err(FilesError::Repeated(word.to_string()));
            }
        }
        Ok(Vocab {
            text: text.into_boxed_str(),
            ends: ends.into_boxed_slice(),
            slots,
        })
    }

    /// `word` 的编号。
    pub(crate) fn get(&self, word: &str) -> Option<u32> {
        self.slots.find(hash_word(word), |id| self.word(id) == word)
    }

    /// 第 `id` 个词。
    fn word(&self, id: u32) -> &str {
        word_at(&self.text, &self.ends, id)
    }

    /// 最长的一个词有几个字。
    pub(crate) fn longest(&self) -> usize {
        (0..self.ends.len())
            .filter_map(|id| u32::try_from(id).ok())
            .map(|id| self.word(id).chars().count())
            .max()
            .unwrap_or(1)
    }
}

/// `text` 里第 `id` 个词（`ends` 记着每个词的结尾，都落在字的边界上：一个个整行接起来的）。
fn word_at<'a>(text: &'a str, ends: &[u32], id: u32) -> &'a str {
    let at = id as usize;
    let start = at.checked_sub(1).map_or(0, |before| ends[before] as usize);
    &text[start..ends[at] as usize]
}

/// 合并：相邻的两个编号能合成哪一个，第几条（从 0 数）就是先后。
#[derive(Debug, Clone)]
pub(crate) struct Merges {
    /// 照先后：左、右、合成的。
    rules: Box<[[u32; 3]]>,
    slots: Slots,
}

impl Merges {
    /// 照先后一条一条放、放得下 `count` 条的一份；条数放不进 `u32` 的交回 `None`。
    pub(crate) fn builder(count: usize) -> Option<MergesBuilder> {
        u32::try_from(count).ok().filter(|n| *n < EMPTY)?;
        Some(MergesBuilder {
            rules: Vec::with_capacity(count),
            limit: count,
            slots: Slots::new(count),
        })
    }

    /// 左 `left`、右 `right` 能合成的：（先后，合成的编号）。
    pub(crate) fn get(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        let rank = self.slots.find(hash_pair(left, right), |rank| {
            let [l, r, _] = self.rules[rank as usize];
            l == left && r == right
        })?;
        Some((rank, self.rules[rank as usize][2]))
    }
}

/// 一条一条照先后放合并。
#[derive(Debug)]
pub(crate) struct MergesBuilder {
    rules: Vec<[u32; 3]>,
    /// 至多放几条。
    limit: usize,
    slots: Slots,
}

impl MergesBuilder {
    /// 放下一条；同一对前面放过的不放、交回 `false`。
    ///
    /// # Panics
    ///
    /// 放的比造的时候说的条数多：表的槽是照那个条数留的，多放会填满、找不到空槽。调用的一方照同一份字数的行。
    pub(crate) fn push(&mut self, left: u32, right: u32, to: u32) -> bool {
        assert!(self.rules.len() < self.limit, "合并比说好的条数多");
        let rank = u32::try_from(self.rules.len()).unwrap_or(EMPTY);
        let rules = &self.rules;
        let fresh = self.slots.insert(hash_pair(left, right), rank, |there| {
            let [l, r, _] = rules[there as usize];
            l == left && r == right
        });
        if fresh {
            self.rules.push([left, right, to]);
        }
        fresh
    }

    /// 造好。
    pub(crate) fn finish(self) -> Merges {
        Merges {
            rules: self.rules.into_boxed_slice(),
            slots: self.slots,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Merges, Vocab};
    use crate::bpe::FilesError;

    #[test]
    fn every_word_is_found_by_its_line_and_others_are_not() {
        // 够多的词，让线性探查真的撞上、绕过去。
        let words: Vec<String> = (0..5000).map(|n| format!("w{n}")).collect();
        let vocab = Vocab::new(&words.join("\n")).expect("没有重复的");
        for (id, word) in (0_u32..).zip(&words) {
            assert_eq!(vocab.get(word), Some(id), "{word}");
        }
        for missing in ["", "w", "w5000", "w00", "x1"] {
            assert_eq!(vocab.get(missing), None, "{missing}");
        }
        assert_eq!(vocab.longest(), 5);
    }

    #[test]
    fn a_repeated_word_is_refused() {
        assert_eq!(
            Vocab::new("a\nb\na").map(|_| ()),
            Err(FilesError::Repeated("a".to_string()))
        );
    }

    #[test]
    fn merges_keep_their_order_and_refuse_a_pair_twice() {
        let mut builder = Merges::builder(4000).expect("放得下");
        for n in 0..4000 {
            assert!(builder.push(n, n + 1, n + 7));
        }
        let merges = builder.finish();
        for n in 0..4000 {
            assert_eq!(merges.get(n, n + 1), Some((n, n + 7)));
            assert_eq!(merges.get(n + 1, n), None);
        }
        let mut builder = Merges::builder(2).expect("放得下");
        assert!(builder.push(1, 2, 3));
        assert!(!builder.push(1, 2, 4));
    }
}

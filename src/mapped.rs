//! 查表用的大权重不拷进匿名内存（EM-4）：把模型文件只读映射进来，找出只给 `Gather` 一类查表算子用的大初始值，造成直接指着
//! 映射那一段的张量，交给 ONNX Runtime 顶替模型里的同名初始值（`AddInitializer`，它不拷、直接用）。
//!
//! 为什么：granite 的 4 位量化模型 61 MB 里，词向量表（`tok_embeddings` 的 4 位权重和比例）占 41 MB。ONNX Runtime 照常载入时
//! 把它拷进匿名内存，常驻不还；可每句话只查几百行。映射以后它算文件页：只有查过的那几页在内存里，内存紧的时候系统自己收回，
//! 要用再从文件读。矩阵乘法的权重不映射：ONNX Runtime 给它们预打包（换成算得快的排法、丢掉原样的那份），顶替了就不打包，
//! 实测短句慢四倍多。
//!
//! 只在小端的 Unix 上做：模型里的数是小端存的，大端的机器要转换、不能直接用；Windows 上映射着的文件删不掉、换不了，包更新时会
//! 撞上。别的平台 [`lookup_tables`] 什么都不找，照原来的载入。

// 这一个文件要 `unsafe`：映射文件、把映射的地址交给 ONNX Runtime 造张量，都没有安全的接口。别处照旧不许。
#![allow(unsafe_code)]
// 别的平台不映射，解析的那几样用不上。
#![cfg_attr(not(all(unix, target_endian = "little")), allow(dead_code))]

use std::path::Path;

use ort::value::DynValue;

/// 至少这么大的初始值才映射：小的省不了几页，不值得多一段映射。
const MIN_BYTES: usize = 1 << 20;

/// 只读查表、不预打包的算子：它们的数据初始值可以换成映射的那一段。
const LOOKUPS: [&str; 4] = [
    "Gather",
    "GatherBlockQuantized",
    "GatherElements",
    "GatherND",
];

/// 带子图的算子：子图里也可能用到外面的初始值，这里不往里看，见到它们就整份不映射。
const SUBGRAPHS: [&str; 3] = ["If", "Loop", "Scan"];

/// 映射着的模型文件。ONNX Runtime 的会话用着它，要比会话活得久（`model.rs` 里它排在会话后面，先放会话）。
pub(crate) struct Mapping {
    #[cfg(all(unix, target_endian = "little"))]
    _map: memmap2::Mmap,
}

impl std::fmt::Debug for Mapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Mapping")
    }
}

/// 顶替用的初始值：名字和指着映射的张量。
pub(crate) type Tables = Vec<(String, DynValue)>;

/// 找到的一个初始值：名字、形状、ONNX 的类型编号，`raw_data` 在文件里的字节范围。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Found {
    pub(crate) name: String,
    pub(crate) dims: Vec<i64>,
    pub(crate) data_type: u64,
    pub(crate) range: std::ops::Range<usize>,
}

/// 映射 `model`，造好顶替用的初始值。文件打不开、映射不了、读不懂、没有可映射的，交回 `None`：照原来的载入，不算错。
///
/// # Errors
///
/// 找到了却造不出张量（ONNX Runtime 不认这个类型、长度）：照原话交回，载入失败，不悄悄退回去拷一份。
#[cfg(all(unix, target_endian = "little"))]
pub(crate) fn lookup_tables(model: &Path) -> Result<Option<(Mapping, Tables)>, ort::Error> {
    let Ok(file) = std::fs::File::open(model) else {
        return Ok(None);
    };
    // 安全：映射是只读的；模型文件是包里的，核心换包时写新文件，不在原地改写这一份。万一被截短，读到截掉的部分进程会收到
    // SIGBUS 退出，核心照常重新拉起。
    let Ok(map) = (unsafe { memmap2::Mmap::map(&file) }) else {
        return Ok(None);
    };
    let found = scan(&map).unwrap_or_default();
    if found.is_empty() {
        return Ok(None);
    }
    let info = ort::memory::MemoryInfo::new(
        ort::memory::AllocationDevice::CPU,
        0,
        ort::memory::AllocatorType::Device,
        ort::memory::MemoryType::Default,
    )?;
    let mut values = Vec::with_capacity(found.len());
    for table in found {
        let Some(element) = element_type(table.data_type) else {
            continue;
        };
        let bytes = &map[table.range.clone()];
        let mut value = std::ptr::null_mut();
        // 安全：指针指着映射的一段，长度照实给；张量只交给 ONNX Runtime 读（映射只读，写一下就是段错误，不会悄悄改坏），
        // 映射比会话活得久。
        unsafe {
            let status = (ort::api().CreateTensorWithDataAsOrtValue)(
                ort::AsPointer::ptr(&info),
                bytes.as_ptr().cast_mut().cast(),
                bytes.len(),
                table.dims.as_ptr(),
                table.dims.len(),
                element,
                &mut value,
            );
            ort::Error::result_from_status(status)?;
        }
        let Some(value) = std::ptr::NonNull::new(value) else {
            continue;
        };
        // 安全：刚造的、不属于哪个会话的张量。
        values.push((table.name, unsafe { DynValue::from_ptr(value, None) }));
    }
    Ok(Some((Mapping { _map: map }, values)))
}

/// 别的平台不映射，照原来的载入。
#[cfg(not(all(unix, target_endian = "little")))]
pub(crate) fn lookup_tables(_model: &Path) -> Result<Option<(Mapping, Tables)>, ort::Error> {
    Ok(None)
}

/// 这几种类型直接用文件里的字节；别的（字符串之类）不碰。
fn element_type(code: u64) -> Option<ort::sys::ONNXTensorElementDataType> {
    use ort::sys::ONNXTensorElementDataType as T;
    Some(match code {
        1 => T::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        2 => T::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        3 => T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        10 => T::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        16 => T::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        21 => T::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
        22 => T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
        _ => return None,
    })
}

/// 一个元素几位。
fn element_bits(code: u64) -> Option<u64> {
    Some(match code {
        1 => 32,
        2 | 3 => 8,
        10 | 16 => 16,
        21 | 22 => 4,
        _ => return None,
    })
}

/// protobuf 的一个字段。
enum Field {
    /// 变长整数。
    Varint(u64),
    /// 带长度的一段：在整份字节里的范围。
    Bytes(std::ops::Range<usize>),
    /// 定长的 4、8 个字节：用不着。
    Fixed,
}

/// 读一个变长整数，`at` 挪到它后面。
fn varint(bytes: &[u8], at: &mut usize) -> Option<u64> {
    let mut value = 0_u64;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(*at)?;
        *at += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte < 0x80 {
            return Some(value);
        }
    }
    None
}

/// `range` 里的一个个字段：（编号，字段）。读不懂的（坏了、用了分组）交回 `None`。
fn fields(bytes: &[u8], range: std::ops::Range<usize>) -> Option<Vec<(u64, Field)>> {
    let mut out = Vec::new();
    let mut at = range.start;
    while at < range.end {
        let key = varint(bytes, &mut at)?;
        let field = match key & 7 {
            0 => Field::Varint(varint(bytes, &mut at)?),
            2 => {
                let len = usize::try_from(varint(bytes, &mut at)?).ok()?;
                let end = at.checked_add(len).filter(|end| *end <= range.end)?;
                let field = Field::Bytes(at..end);
                at = end;
                field
            }
            1 => {
                at += 8;
                Field::Fixed
            }
            5 => {
                at += 4;
                Field::Fixed
            }
            _ => return None,
        };
        out.push((key >> 3, field));
    }
    (at == range.end).then_some(out)
}

fn text(bytes: &[u8], range: &std::ops::Range<usize>) -> Option<String> {
    String::from_utf8(bytes[range.clone()].to_vec()).ok()
}

/// 在一份 ONNX 模型（`ModelProto`）的字节里找可以映射的初始值：至少 [`MIN_BYTES`]、数据就在 `raw_data` 里、形状和长度对得上、
/// 只给 [`LOOKUPS`] 用的、不是图的输入。读不懂的交回 `None`；有带子图的算子的交回空的。
pub(crate) fn scan(bytes: &[u8]) -> Option<Vec<Found>> {
    // ModelProto.graph = 7；GraphProto.node = 1、initializer = 5、input = 11；NodeProto.input = 1、op_type = 4；
    // TensorProto.dims = 1、data_type = 2、name = 8、raw_data = 9、data_location = 14，别的数据字段 4～7、10、11。
    let mut graphs = fields(bytes, 0..bytes.len())?
        .into_iter()
        .filter_map(|(no, field)| match (no, field) {
            (7, Field::Bytes(range)) => Some(range),
            _ => None,
        });
    let graph = graphs.next()?;
    if graphs.next().is_some() {
        return None;
    }
    let mut lookups_only = std::collections::HashMap::<String, bool>::new();
    let mut inputs = std::collections::HashSet::new();
    let mut tables = Vec::new();
    for (no, field) in fields(bytes, graph)? {
        let Field::Bytes(range) = field else { continue };
        match no {
            1 => {
                let (mut op, mut names) = (String::new(), Vec::new());
                for (no, field) in fields(bytes, range)? {
                    match (no, field) {
                        (1, Field::Bytes(name)) => names.push(text(bytes, &name)?),
                        (4, Field::Bytes(name)) => op = text(bytes, &name)?,
                        _ => {}
                    }
                }
                if SUBGRAPHS.contains(&op.as_str()) {
                    return Some(Vec::new());
                }
                // 查表算子按行读的输入：第一个是表；`GatherBlockQuantized` 的比例、零点（第三、第四个）也按同样的行读。
                // 下标（第二个）和别的算子的输入都照「不是表」记。
                for (at, name) in names.into_iter().enumerate() {
                    let lookup = LOOKUPS.contains(&op.as_str())
                        && (at == 0 || (op == "GatherBlockQuantized" && at >= 2));
                    let only = lookups_only.entry(name).or_insert(true);
                    *only &= lookup;
                }
            }
            5 => tables.push(range),
            11 => {
                for (no, field) in fields(bytes, range)? {
                    if let (1, Field::Bytes(name)) = (no, field) {
                        inputs.insert(text(bytes, &name)?);
                    }
                }
            }
            _ => {}
        }
    }
    let mut found = Vec::new();
    for range in tables {
        let (mut dims, mut data_type, mut name, mut raw, mut other) =
            (Vec::new(), 0, None, None, false);
        for (no, field) in fields(bytes, range)? {
            match (no, field) {
                (1, Field::Varint(dim)) => dims.push(i64::try_from(dim).ok()?),
                (1, Field::Bytes(packed)) => {
                    let mut at = packed.start;
                    while at < packed.end {
                        dims.push(i64::try_from(varint(bytes, &mut at)?).ok()?);
                    }
                }
                (2, Field::Varint(code)) => data_type = code,
                (8, Field::Bytes(range)) => name = Some(text(bytes, &range)?),
                (9, Field::Bytes(range)) => raw = Some(range),
                (14, Field::Varint(location)) => other |= location != 0,
                (4..=7 | 10 | 11 | 13, _) => other = true,
                _ => {}
            }
        }
        let (Some(name), Some(raw)) = (name, raw) else {
            continue;
        };
        let count = dims.iter().try_fold(1_u64, |count, dim| {
            count.checked_mul(u64::try_from(*dim).ok()?)
        });
        let want = count
            .zip(element_bits(data_type))
            .and_then(|(count, bits)| count.checked_mul(bits))
            .map(|bits| bits.div_ceil(8));
        if other
            || raw.len() < MIN_BYTES
            || want != u64::try_from(raw.len()).ok()
            || inputs.contains(&name)
            || lookups_only.get(&name) != Some(&true)
            || found.iter().any(|table: &Found| table.name == name)
        {
            continue;
        }
        found.push(Found {
            name,
            dims,
            data_type,
            range: raw,
        });
    }
    Some(found)
}

#[cfg(test)]
mod tests {
    use super::{Found, MIN_BYTES, scan};

    fn varint(mut n: u64, out: &mut Vec<u8>) {
        while n >= 0x80 {
            out.push((n as u8) | 0x80);
            n >>= 7;
        }
        out.push(n as u8);
    }

    fn bytes(no: u64, data: &[u8], out: &mut Vec<u8>) {
        varint(no << 3 | 2, out);
        varint(data.len() as u64, out);
        out.extend_from_slice(data);
    }

    fn number(no: u64, n: u64, out: &mut Vec<u8>) {
        varint(no << 3, out);
        varint(n, out);
    }

    fn node(op: &str, inputs: &[&str]) -> Vec<u8> {
        let mut out = Vec::new();
        for input in inputs {
            bytes(1, input.as_bytes(), &mut out);
        }
        bytes(4, op.as_bytes(), &mut out);
        out
    }

    /// 一个 float 初始值，`floats` 个数，`raw_data` 全是 0。
    fn tensor(name: &str, floats: u64) -> Vec<u8> {
        let mut out = Vec::new();
        number(1, floats, &mut out);
        number(2, 1, &mut out);
        bytes(8, name.as_bytes(), &mut out);
        bytes(9, &vec![0; floats as usize * 4], &mut out);
        out
    }

    fn model(nodes: &[Vec<u8>], tensors: &[Vec<u8>]) -> Vec<u8> {
        let mut graph = Vec::new();
        for node in nodes {
            bytes(1, node, &mut graph);
        }
        for tensor in tensors {
            bytes(5, tensor, &mut graph);
        }
        let mut out = Vec::new();
        number(1, 9, &mut out);
        bytes(7, &graph, &mut out);
        out
    }

    const BIG: u64 = (MIN_BYTES / 4) as u64;

    #[test]
    fn only_a_big_table_that_only_lookups_read_is_mapped() {
        let bytes = model(
            &[
                node("Gather", &["table", "ids"]),
                node("MatMul", &["x", "weight"]),
                node("Gather", &["shared", "ids"]),
                node("Add", &["shared", "x"]),
                node("Gather", &["small", "ids"]),
                node("GatherBlockQuantized", &["q4", "big_ids", "scales"]),
            ],
            &[
                tensor("table", BIG),
                tensor("weight", BIG),
                tensor("shared", BIG),
                tensor("small", 16),
                tensor("q4", BIG),
                tensor("big_ids", BIG),
                tensor("scales", BIG),
            ],
        );
        let found = scan(&bytes).expect("读得懂");
        let names: Vec<&str> = found.iter().map(|table| table.name.as_str()).collect();
        assert_eq!(names, ["table", "q4", "scales"]);
        let Found {
            dims,
            data_type,
            range,
            ..
        } = &found[0];
        assert_eq!((dims.as_slice(), *data_type), (&[BIG as i64][..], 1));
        assert_eq!(range.len(), MIN_BYTES);
        assert!(bytes[range.clone()].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn a_subgraph_or_a_broken_file_maps_nothing() {
        let with_if = model(
            &[node("Gather", &["table", "ids"]), node("If", &["cond"])],
            &[tensor("table", BIG)],
        );
        assert_eq!(scan(&with_if), Some(Vec::new()));
        let whole = model(
            &[node("Gather", &["table", "ids"])],
            &[tensor("table", BIG)],
        );
        assert_eq!(scan(&whole[..whole.len() - 1]), None);
        assert_eq!(scan(b"not a model"), None);
    }
}

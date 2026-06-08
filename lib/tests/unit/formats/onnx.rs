use super::*;

#[test]
fn test_block_index_onnx_layer() {
    assert_eq!(block_index_from_name_onnx("layer.0.attention.self.query"), Some(0));
    assert_eq!(block_index_from_name_onnx("layer.11.output.dense"), Some(11));
}

#[test]
fn test_block_index_onnx_transformer() {
    assert_eq!(
        block_index_from_name_onnx("transformer.h.5.attn.c_attn.weight"),
        Some(5)
    );
}

#[test]
fn test_block_index_onnx_bert() {
    assert_eq!(
        block_index_from_name_onnx("encoder.layer.3.intermediate.dense"),
        Some(3)
    );
}

#[test]
fn test_block_index_onnx_none() {
    assert_eq!(block_index_from_name_onnx("embedding.word"), None);
    assert_eq!(block_index_from_name_onnx("lm_head.weight"), None);
}

#[test]
fn test_onnx_dtype_mapping() {
    assert_eq!(onnx_dtype_to_tensor(1), Some(TensorDtype::F32));
    assert_eq!(onnx_dtype_to_tensor(10), Some(TensorDtype::F16));
    assert_eq!(onnx_dtype_to_tensor(16), Some(TensorDtype::Bf16));
    assert_eq!(onnx_dtype_to_tensor(3), Some(TensorDtype::I8));
    assert_eq!(onnx_dtype_to_tensor(6), Some(TensorDtype::I32));
    assert_eq!(onnx_dtype_to_tensor(999), None);
}

#[test]
fn test_byte_size_for_dtype() {
    assert_eq!(byte_size_for_dtype(1), Some(4));
    assert_eq!(byte_size_for_dtype(10), Some(2));
    assert_eq!(byte_size_for_dtype(3), Some(1));
    assert_eq!(byte_size_for_dtype(11), Some(8));
}

#[test]
fn test_minimal_onnx_roundtrip() {
    let proto = ModelProto {
        ir_version: 9,
        producer_name: "test-producer".into(),
        producer_version: "1.0".into(),
        domain: "test".into(),
        model_version: 1,
        doc_string: String::new(),
        opset_import: vec![OperatorSetIdProto {
            domain: "".into(),
            version: 21,
        }],
        metadata_props: vec![StringStringEntryProto {
            key: "architecture".into(),
            value: "bert".into(),
        }],
        graph: Some(GraphProto {
            name: "test-graph".into(),
            input: vec![ValueInfoProto {
                name: "input_ids".into(),
                doc_string: String::new(),
            }],
            output: vec![ValueInfoProto {
                name: "logits".into(),
                doc_string: String::new(),
            }],
            initializer: vec![
                TensorProto {
                    name: "layer.0.attention.weight".into(),
                    data_type: 1,
                    dims: vec![64, 64],
                    raw_data: vec![0u8; 64 * 64 * 4],
                    ..Default::default()
                },
                TensorProto {
                    name: "layer.0.output.dense.bias".into(),
                    data_type: 1,
                    dims: vec![64],
                    raw_data: vec![0u8; 64 * 4],
                    ..Default::default()
                },
                TensorProto {
                    name: "embedding.word_embeddings".into(),
                    data_type: 1,
                    dims: vec![1000, 64],
                    raw_data: vec![0u8; 1000 * 64 * 4],
                    ..Default::default()
                },
            ],
        }),
    };

    let bytes = proto.encode_to_vec();
    let onnx = OnnxFile::from_bytes(&bytes).expect("parse onnx");

    assert_eq!(onnx.format(), ModelFormat::Onnx);
    assert_eq!(onnx.name(), Some("test-producer"));
    assert_eq!(onnx.architecture(), Some("bert"));
    assert_eq!(onnx.tensors.len(), 3);
    assert_eq!(onnx.block_count(), Some(1));

    let t = onnx.tensor("layer.0.attention.weight").unwrap();
    assert_eq!(t.shape, vec![64, 64]);

    let input_names = onnx.input_names();
    assert!(input_names.contains(&"input_ids"));
    let output_names = onnx.output_names();
    assert!(output_names.contains(&"logits"));
}

#[test]
fn test_write_basic_roundtrip() {
    let mut w = OnnxWriter::new();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    w.add_raw("weight", 1, &[4, 4], &bytes);

    let out = w.into_bytes();
    let onnx = OnnxFile::from_bytes(&out).expect("parse back");

    assert_eq!(onnx.format(), ModelFormat::Onnx);
    assert_eq!(onnx.tensors.len(), 1);
    let t = onnx.tensor("weight").unwrap();
    assert_eq!(t.shape, vec![4, 4]);
    assert_eq!(t.dtype, TensorDtype::F32);
    assert_eq!(t.byte_size, 64);
}

#[test]
fn test_write_with_metadata() {
    let mut w = OnnxWriter::new()
        .producer("test-producer", "2.0")
        .graph_name("my-graph");
    w.add_metadata("architecture", "bert");
    w.add_metadata("model_type", "encoder");

    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    w.add_raw("bias", 1, &[3], &bytes);

    let out = w.into_bytes();
    let onnx = OnnxFile::from_bytes(&out).expect("parse back");

    assert_eq!(onnx.name(), Some("test-producer"));
    assert_eq!(onnx.architecture(), Some("bert"));
    assert_eq!(onnx.tensors.len(), 1);
}

#[test]
fn test_write_multiple_dtypes() {
    let mut w = OnnxWriter::new();

    let f32_data: Vec<f32> = vec![1.0, 2.0];
    w.add_raw("f32_tensor", onnx_dtypes::FLOAT, &[2],
        &f32_data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());

    let i64_data: Vec<i64> = vec![100, 200];
    w.add_raw("i64_tensor", onnx_dtypes::INT64, &[2],
        &i64_data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());

    let f16_bytes: Vec<u8> = vec![0, 60, 0, 64];
    w.add_raw("f16_tensor", onnx_dtypes::FLOAT16, &[2], &f16_bytes);

    let out = w.into_bytes();
    let onnx = OnnxFile::from_bytes(&out).expect("parse back");

    assert_eq!(onnx.tensors.len(), 3);
    assert_eq!(onnx.tensor("f32_tensor").unwrap().dtype, TensorDtype::F32);
    assert_eq!(onnx.tensor("i64_tensor").unwrap().dtype, TensorDtype::I64);
    assert_eq!(onnx.tensor("f16_tensor").unwrap().dtype, TensorDtype::F16);
}

#[test]
fn test_write_empty_tensors() {
    let w = OnnxWriter::new();
    let out = w.into_bytes();
    let onnx = OnnxFile::from_bytes(&out).expect("parse back");
    assert_eq!(onnx.tensors.len(), 0);
}

#[test]
fn test_write_graph_inputs_outputs() {
    let mut w = OnnxWriter::new().graph_name("test-graph");
    w.add_raw("weight", onnx_dtypes::FLOAT, &[2, 2], &vec![0u8; 16]);
    w.add_graph_input("input_ids", &[1, 128]);
    w.add_graph_input("attention_mask", &[1, 128]);
    w.add_graph_output("logits", &[1, 1000]);

    let out = w.into_bytes();
    let onnx = OnnxFile::from_bytes(&out).expect("parse back");

    assert_eq!(onnx.tensors.len(), 1);
    let inputs = onnx.input_names();
    assert!(inputs.contains(&"input_ids"));
    assert!(inputs.contains(&"attention_mask"));
    let outputs = onnx.output_names();
    assert!(outputs.contains(&"logits"));
}

#[test]
fn test_write_to_file() {
    let dir = std::env::temp_dir();
    let path = dir.join("test_write_onnx.onnx");
    let mut w = OnnxWriter::new();
    let data: Vec<f32> = vec![0.5; 64];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    w.add_raw("matrix", onnx_dtypes::FLOAT, &[8, 8], &bytes);

    let f = std::fs::File::create(&path).expect("create temp file");
    w.write_to(f).expect("write onnx");

    let meta = std::fs::metadata(&path).expect("metadata");
    assert!(meta.len() > 0);

    let onnx = OnnxFile::open(&path).expect("read back");
    assert_eq!(onnx.tensors.len(), 1);
    let t = onnx.tensor("matrix").unwrap();
    assert_eq!(t.shape, vec![8, 8]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_tensor_dtype_to_onnx_mapping() {
    assert_eq!(tensor_dtype_to_onnx(TensorDtype::F32), Some(1));
    assert_eq!(tensor_dtype_to_onnx(TensorDtype::F16), Some(10));
    assert_eq!(tensor_dtype_to_onnx(TensorDtype::Bf16), Some(16));
    assert_eq!(tensor_dtype_to_onnx(TensorDtype::I8), Some(3));
    assert_eq!(tensor_dtype_to_onnx(TensorDtype::I32), Some(6));
    assert_eq!(tensor_dtype_to_onnx(TensorDtype::Unknown(0)), Some(9));
    assert_eq!(tensor_dtype_to_onnx(TensorDtype::Q4_0), None);
    assert_eq!(tensor_dtype_to_onnx(TensorDtype::Iq2S), None);
}

#[test]
fn test_add_tensor_via_tensordtype() {
    let mut w = OnnxWriter::new();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    w.add_tensor("weights", TensorDtype::F32, &[2, 2], &bytes).expect("add_tensor");

    let out = w.into_bytes();
    let onnx = OnnxFile::from_bytes(&out).expect("parse");
    assert_eq!(onnx.tensors.len(), 1);
    assert_eq!(onnx.tensor("weights").unwrap().dtype, TensorDtype::F32);
}

#[test]
fn test_add_tensor_unsupported_dtype_errors() {
    let mut w = OnnxWriter::new();
    let result = w.add_tensor("q", TensorDtype::Q4_0, &[4, 4], &vec![0u8; 32]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("unsupported TensorDtype"));
}

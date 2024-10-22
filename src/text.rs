use anyhow::Result;
use ndarray::{Array, ArrayView};
use ort::{GraphOptimizationLevel, Session};
use std::collections::HashMap;
use tokenizers::Tokenizer;

pub fn get_text_feature(text: &str, model_path: &str, tokenizer_path: &str) -> Result<Vec<f32>> {
    // 加载ONNX模型
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    // 使用tokenizer对文本进行编码
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let encoding = tokenizer.encode(text, true).unwrap();
    let mut input_ids = encoding.get_ids().to_vec();
    input_ids.resize(52, 0); // 填充到52个token

    // 准备输入tensor
    let input_tensor = Array::from_shape_vec((1, 52), input_ids)?
        .map(|&x| x as i64);

    // 运行推理
    let input_value = ort::Value::from_array(input_tensor.into_dyn())?;
    let mut inputs = HashMap::new();
    inputs.insert("text".to_string(), input_value);

    let outputs = session.run(inputs)?;

    // 获取输出并归一化
    let features: ArrayView<f32, _> = outputs[0].try_extract_tensor()?;
    let mut features_vec: Vec<f32> = features.to_owned().into_raw_vec_and_offset().0;
    let norm: f32 = features_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    features_vec.iter_mut().for_each(|x| *x /= norm);

    Ok(features_vec)
}

use anyhow::Result;
use ndarray::{Array, ArrayView};
use ort::{GraphOptimizationLevel, Session};
use std::collections::HashMap;
use tokenizers::Tokenizer;

pub struct TextProcessor {
    session: Session,
    tokenizer: Tokenizer,
    max_length: usize,
}

impl TextProcessor {
    pub fn new(model_path: &str, tokenizer_path: &str, max_length: usize) -> Result<Self> {
        // 加载ONNX模型
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        // 加载tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("加载tokenizer失败: {}", e))?;

        Ok(Self {
            session,
            tokenizer,
            max_length,
        })
    }

    pub fn process_text(&self, text: &str) -> Result<Vec<f32>> {
        // 使用tokenizer对文本进行编码
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("编码文本失败: {}", e))?;

        let mut input_ids = encoding.get_ids().to_vec();
        input_ids.resize(self.max_length, 0); // 填充到指定长度

        // 准备输入tensor
        let input_tensor = Array::from_shape_vec((1, self.max_length), input_ids)?
            .map(|&x| x as i64);

        // 运行推理
        let input_value = ort::Value::from_array(input_tensor.into_dyn())?;
        let mut inputs = HashMap::new();
        inputs.insert("text".to_string(), input_value);

        let outputs = self.session.run(inputs)?;

        // 获取输出并归一化
        let features: ArrayView<f32, _> = outputs[0].try_extract_tensor()?;
        let mut features_vec: Vec<f32> = features.to_owned().into_raw_vec_and_offset().0;
        let norm: f32 = features_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        features_vec.iter_mut().for_each(|x| *x /= norm);

        Ok(features_vec)
    }
}

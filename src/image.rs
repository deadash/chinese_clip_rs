use std::collections::HashMap;

use anyhow::Result;
use ndarray::{Array, ArrayView};
use ort::{GraphOptimizationLevel, Session};

pub fn get_image_feature(image_path: &str) -> Result<Vec<f32>> {
    // 加载ONNX模型
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("models/clip_cn_vit-l-14.img.fp32.onnx")?;

    // 读取并预处理图片
    let img = image::ImageReader::open(image_path)?.decode()?;
    let img = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
    let img = img.to_rgb8();

    // 转换为张量并归一化
    let mut img_tensor = Array::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let pixel = img.get_pixel(x as u32, y as u32);
        pixel[c] as f32 / 255.0
    });

    // 应用均值和标准差归一化
    let mean = Array::from_shape_vec((1, 3, 1, 1), vec![0.48145466, 0.4578275, 0.40821073])?;
    let std = Array::from_shape_vec((1, 3, 1, 1), vec![0.26862954, 0.26130258, 0.27577711])?;
    img_tensor = (img_tensor - mean) / std;

    // 运行推理
    let input_value = ort::Value::from_array(img_tensor.into_dyn())?;
    let mut inputs = HashMap::new();
    inputs.insert("image".to_string(), input_value);

    let outputs = session.run(inputs)?;

    // 获取输出并归一化
    let features: ArrayView<f32, _> = outputs[0].try_extract_tensor()?;
    let mut features_vec: Vec<f32> = features.to_owned().into_raw_vec_and_offset().0;
    let norm: f32 = features_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    features_vec.iter_mut().for_each(|x| *x /= norm);

    Ok(features_vec)
}
use anyhow::Result;
use ndarray::{Array, ArrayView, Ix4};
use ort::{GraphOptimizationLevel, Session};
use std::collections::HashMap;
use image::{DynamicImage, ImageBuffer, Rgb};

pub struct ImageProcessor {
    session: Session,
    resolution: (u32, u32),
}

impl ImageProcessor {
    pub fn new(model_path: &str, resolution: (u32, u32)) -> Result<Self> {
        // 加载ONNX模型
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self { session, resolution })
    }

    pub fn process_image<T: AsRef<[u8]>>(&self, image_data: T) -> Result<Vec<f32>> {
        let img = image::load_from_memory(image_data.as_ref())?;
        self.process_dynamic_image(&img)
    }

    pub fn process_file(&self, image_path: &str) -> Result<Vec<f32>> {
        let img = image::open(image_path)?;
        self.process_dynamic_image(&img)
    }

    fn process_dynamic_image(&self, img: &DynamicImage) -> Result<Vec<f32>> {
        let (width, height) = self.resolution;

        // 调整图像大小并转换为RGB
        let img = img.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
        let img = img.to_rgb8();

        // 转换为张量并归一化
        let img_tensor = self.preprocess_image(&img)?;

        // 运行推理
        let input_value = ort::Value::from_array(img_tensor.into_dyn())?;
        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), input_value);

        let outputs = self.session.run(inputs)?;

        // 获取输出并归一化
        let features: ArrayView<f32, _> = outputs[0].try_extract_tensor()?;
        let mut features_vec: Vec<f32> = features.to_owned().into_raw_vec_and_offset().0;
        let norm: f32 = features_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        features_vec.iter_mut().for_each(|x| *x /= norm);

        Ok(features_vec)
    }

    fn preprocess_image(&self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Array<f32, Ix4>> {
        let (width, height) = self.resolution;

        // 转换为张量并归一化
        let mut img_tensor = Array::from_shape_fn((1, 3, height as usize, width as usize), |(_, c, y, x)| {
            let pixel = img.get_pixel(x as u32, y as u32);
            pixel[c] as f32 / 255.0
        });

        // 应用均值和标准差归一化
        let mean = Array::from_shape_vec((1, 3, 1, 1), vec![0.48145466, 0.4578275, 0.40821073])?;
        let std = Array::from_shape_vec((1, 3, 1, 1), vec![0.26862954, 0.26130258, 0.27577711])?;
        img_tensor = (img_tensor - mean) / std;

        Ok(img_tensor)
    }
}

use anyhow::Result;
use chinese_clip_rs::{calculate_similarity, softmax, ImageProcessor, TextProcessor};
use std::time::Instant;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut providers = Vec::new();

    #[cfg(feature = "tensorrt")]
    {
        use ort::TensorRTExecutionProvider;
        providers.push(TensorRTExecutionProvider::default().build());
    }

    #[cfg(feature = "cuda")]
    {
        use ort::{ExecutionProvider, CUDAExecutionProvider};
        let provider = CUDAExecutionProvider::default();
        // if provider.supported_by_platform() {
        //     println!("CUDA 支持");
        // }
        providers.push(provider.build());
    }

    #[cfg(feature = "openvino")]
    {
        use ort::OpenVINOExecutionProvider;
        providers.push(OpenVINOExecutionProvider::default().build());
    }

    #[cfg(feature = "directml")]
    {
        use ort::DirectMLExecutionProvider;
        providers.push(DirectMLExecutionProvider::default().build());
    }

    ort::init()
        .with_execution_providers(providers)
        .commit()?;


    let image_path = "tests/pokemon.jpeg";
    let image_model_path = "models/clip_cn_vit-l-14.img.fp32.onnx";
    let text_model_path = "models/clip_cn_vit-l-14.txt.fp32.onnx";
    let tokenizer_path = "models/clip_cn_tokenizer.json";

    let start_time = Instant::now();

    // 创建 ImageProcessor 实例
    let image_processor = ImageProcessor::new(image_model_path, (224, 224))?;
    let image_features = image_processor.process_file(image_path)?;
    
    let image_time = start_time.elapsed();
    tracing::info!("获取图像特征耗时: {:?}", image_time);

    // 创建 TextProcessor 实例
    let text_processor = TextProcessor::new(text_model_path, tokenizer_path, 52)?;

    let pokemon_names = vec!["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"];
    
    let mut all_logits = Vec::new();
    for name in &pokemon_names {
        let text_start_time = Instant::now();
        let text_features = text_processor.process_text(name)?;
        let text_time = text_start_time.elapsed();
        tracing::info!("获取文本特征 '{}' 耗时: {:?}", name, text_time);

        let similarity = calculate_similarity(&image_features, &text_features);
        all_logits.push(similarity);
    }

    let probabilities = softmax(&all_logits);

    for (name, prob) in pokemon_names.iter().zip(probabilities.iter()) {
        tracing::info!("{} 的相似度概率: {:.4}", name, prob);
    }

    Ok(())
}

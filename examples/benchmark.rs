use anyhow::Result;
use chinese_clip_rs::ImageProcessor;
use std::time::{Instant, Duration};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    provider: Option<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let mut providers = Vec::new();

    match args.provider.as_deref() {
        Some("tensorrt") => {
            #[cfg(feature = "tensorrt")]
            {
                use ort::TensorRTExecutionProvider;
                providers.push(TensorRTExecutionProvider::default().build());
            }
        }
        Some("cuda") => {
            #[cfg(feature = "cuda")]
            {
                use ort::CUDAExecutionProvider;
                providers.push(CUDAExecutionProvider::default().build());
            }
        }
        Some("openvino") => {
            #[cfg(feature = "openvino")]
            {
                use ort::OpenVINOExecutionProvider;
                providers.push(OpenVINOExecutionProvider::default().build());
            }
        }
        Some("directml") => {
            #[cfg(feature = "directml")]
            {
                use ort::DirectMLExecutionProvider;
                providers.push(DirectMLExecutionProvider::default().build());
            }
        }
        None => {
            // 如果没有指定provider,则使用所有可用的provider
            // ... (保留原有的所有provider代码)
        }
        Some(unknown) => {
            return Err(anyhow::anyhow!("未知的执行提供程序: {}", unknown));
        }
    }

    ort::init()
        .with_execution_providers(providers)
        .commit()?;

    let image_path = "tests/pokemon.jpeg";
    let image_model_path = "models/clip_cn_vit-l-14.img.fp32.onnx";

    let image_processor = ImageProcessor::new(image_model_path, (224, 224))?;

    let iterations = 100;
    let mut total_duration = Duration::new(0, 0);

    for i in 1..=iterations {
        let start_time = Instant::now();

        let _image_features = image_processor.process_file(image_path)?;

        let iteration_duration = start_time.elapsed();
        total_duration += iteration_duration;

        tracing::info!("迭代 {} 耗时: {:?}", i, iteration_duration);
    }

    let average_duration = total_duration / iterations;
    tracing::info!("平均耗时 ({} 次迭代): {:?}", iterations, average_duration);

    Ok(())
}

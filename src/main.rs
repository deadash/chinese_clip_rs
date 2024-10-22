mod image;
mod text;
use anyhow::Result;

fn calculate_similarity(image_features: &[f32], text_features: &[f32]) -> Vec<f32> {
    assert_eq!(image_features.len(), text_features.len(), "向量长度必须相同");

    // 计算点积
    let logits = image_features.iter()
        .zip(text_features.iter())
        .map(|(&a, &b)| a * b)
        .sum::<f32>() * 100.0;

    // 返回包含单个元素的向量
    vec![logits]
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    logits.iter().map(|&x| ((x - max_logit).exp()) / exp_sum).collect()
}

fn main() -> Result<()> {
    let image_features = image::get_image_feature("tests/pokemon.jpeg")?;
    println!("图片特征向量: {:?}", image_features);

    let pokemon_names = vec!["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"];
    
    let mut all_logits = Vec::new();
    for name in &pokemon_names {
        let text_features = text::get_text_feature(name)?;
        let logits = calculate_similarity(&image_features, &text_features);
        all_logits.extend_from_slice(&logits);
    }

    let probabilities = softmax(&all_logits);

    for (name, prob) in pokemon_names.iter().zip(probabilities.iter()) {
        println!("{} 的相似度概率: {:.4}", name, prob);
    }

    Ok(())
}

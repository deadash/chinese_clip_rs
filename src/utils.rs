pub fn calculate_similarity(image_features: &[f32], text_features: &[f32]) -> f32 {
    assert_eq!(image_features.len(), text_features.len(), "向量长度必须相同");

    // 计算点积
    image_features.iter()
        .zip(text_features.iter())
        .map(|(&a, &b)| a * b)
        .sum::<f32>() * 100.0
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    logits.iter().map(|&x| ((x - max_logit).exp()) / exp_sum).collect()
}


# Chinese CLIP 模型部署

本项目使用ONNX来部署Chinese CLIP模型。

## 模型下载

模型可以从[ModelScope](https://modelscope.cn/collections/chinese_clip-6381f8ddaaf54c)下载。请将下载的模型文件保存到项目的`models`目录中。

## 编译

使用以下命令编译项目:

```bash
cargo build --release --example benchmark --features "tensorrt cuda openvino directml"
```

## 环境配置

### CUDA

- 安装CUDA和cuDNN
- 将cuDNN的bin目录添加到系统PATH环境变量中

### TensorRT

- 下载并解压TensorRT
- 将TensorRT的bin目录添加到系统PATH环境变量中

### DirectML

1. 从[NuGet](https://www.nuget.org/packages/Microsoft.AI.DirectML)下载DirectML包
2. 使用7-Zip解压包
3. 将对应系统版本的DLL文件放置于可执行文件路径或添加到系统PATH中

### OpenVINO

(正在研究中)

## 性能对比

### CPU

| 处理器 | 处理时间 | CPU负载 |
|--------|----------|---------|
| Intel 8505 | 852ms | 100% |
| i7-8700k | 480ms | 46% |

### GPU

| GPU | 后端 | 处理时间 | GPU负载 | 备注 |
|-----|------|----------|---------|------|
| RTX 3090 | CUDA | 14ms | 1% | - |
| RTX 3090 | TensorRT | 10ms | 1% | 3D满载 |
| Intel UHD Graphics (8505集显) | directml | 760ms | - | 3D满载 |


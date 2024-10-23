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

使用OpenVINO需要进行以下步骤:

1. 在编译时启用`load-dynamic`特性:

   ```bash
   cargo build --release --example benchmark --features "openvino load-dynamic"
   ```

2. 下载OpenVINO运行时:
   - 从[Intel OpenVINO官方文档](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_windows.html)按照指示安装OpenVINO运行时
   - 设置必要的环境变量

3. 下载ONNX Runtime OpenVINO执行提供程序:
   - 从[ONNX Runtime Releases](https://github.com/intel/onnxruntime/releases)下载`Microsoft.ML.OnnxRuntime.OpenVino.*.zip`
   - 解压下载的zip文件
   - 再次解压得到的nuget包,找到bin目录下的DLL文件

4. 将解压得到的DLL文件放置于可执行文件同目录或添加到系统PATH中

注意:
- 不建议在GPU上使用OpenVINO,因为性能可能不如其他后端(directml), 包括fp16也没有加速效果
- 在CPU上,OpenVINO可以提供一定的性能优化

使用OpenVINO运行benchmark:

```bash
cargo run --release --example benchmark --features "openvino" -- --provider openvino
```

如果要指定使用GPU:

```bash
cargo run --release --example benchmark --features "openvino" -- --provider openvino --openvino-device GPU
```

## 性能对比

### CPU

| 处理器 | 处理时间 | CPU负载 |
|--------|----------|---------|
| Intel 8505 | 852ms | 100% |
| Intel 8505(openvino) | 830ms | 100% |
| i7-8700k | 480ms | 46% |
| i7-8700k(openvino) | 370ms | 65% |

### GPU

| GPU | 后端 | 处理时间 | GPU负载 | 备注 |
|-----|------|----------|---------|------|
| RTX 3090 | TensorRT | 10ms | 1% | 3D满载 |
| RTX 3090 | CUDA | 14ms | 1% | - |
| RTX 3090 | directml | 22ms | - | - |
| GTX 970M | directml | 480ms | 100% | - |
| Intel UHD Graphics (8505集显) | directml | 760ms | - | 3D满载 |
| Intel UHD Graphics (8505集显) | openvino | 1.34s | - | 3D满载/CPU接近满载 |


#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kINFO) {
      std::cerr << "TensorRT: " << message << '\n';
    }
  }
};

template <typename T>
struct DeleteInferObject {
  void operator()(T* value) const noexcept { delete value; }
};

template <typename T>
using InferPtr = std::unique_ptr<T, DeleteInferObject<T>>;

struct Arguments {
  std::string onnx;
  std::string output;
  std::uint64_t workspace_mib = 2048;
  bool fp16 = true;
};

Arguments parseArguments(int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&]() {
      if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + argument);
      }
      return std::string(argv[++index]);
    };
    if (argument == "--onnx") result.onnx = value();
    else if (argument == "--output") result.output = value();
    else if (argument == "--workspace-mib") {
      result.workspace_mib = std::stoull(value());
    } else if (argument == "--fp16") result.fp16 = true;
    else if (argument == "--fp32") result.fp16 = false;
    else throw std::runtime_error("unknown argument: " + argument);
  }
  if (result.onnx.empty() || result.output.empty()) {
    throw std::runtime_error(
        "usage: autoaim_research_build_engine --onnx MODEL.onnx "
        "--output MODEL.engine [--fp16|--fp32] [--workspace-mib N]");
  }
  if (result.workspace_mib == 0 || result.workspace_mib > 8192) {
    throw std::runtime_error("workspace MiB must be in [1,8192]");
  }
  return result;
}

std::string dimensions(const nvinfer1::Dims& shape) {
  std::string result = "[";
  for (int index = 0; index < shape.nbDims; ++index) {
    if (index != 0) result += ',';
    result += std::to_string(shape.d[index]);
  }
  return result + ']';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parseArguments(argc, argv);
    if (!std::filesystem::is_regular_file(arguments.onnx)) {
      throw std::runtime_error("ONNX model does not exist: " + arguments.onnx);
    }
    const std::filesystem::path output(arguments.output);
    const std::filesystem::path partial = output.string() + ".partial";
    if (std::filesystem::exists(output) || std::filesystem::exists(partial)) {
      throw std::runtime_error(
          "refusing to overwrite protected TensorRT engine: " +
          output.string());
    }
    if (!output.parent_path().empty() &&
        !std::filesystem::is_directory(output.parent_path())) {
      throw std::runtime_error("output directory does not exist: " +
                               output.parent_path().string());
    }

    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess) {
      throw std::runtime_error("cannot query CUDA device 0");
    }
    Logger logger;
    InferPtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    if (!builder) throw std::runtime_error("createInferBuilder failed");
    InferPtr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(0U)};
    if (!network) throw std::runtime_error("createNetworkV2 failed");
    InferPtr<nvonnxparser::IParser> parser{
        nvonnxparser::createParser(*network, logger)};
    if (!parser) throw std::runtime_error("create ONNX parser failed");
    if (!parser->parseFromFile(
            arguments.onnx.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
      std::string message = "ONNX parse failed";
      for (int index = 0; index < parser->getNbErrors(); ++index) {
        message += "\n  ";
        message += parser->getError(index)->desc();
      }
      throw std::runtime_error(message);
    }
    if (network->getNbInputs() != 1 || network->getInput(0) == nullptr) {
      throw std::runtime_error("ONNX network must expose exactly one input");
    }
    const auto expected_input_type = arguments.fp16
        ? nvinfer1::DataType::kHALF
        : nvinfer1::DataType::kFLOAT;
    if (network->getInput(0)->getType() != expected_input_type) {
      throw std::runtime_error(
          arguments.fp16
              ? "TensorRT 11 FP16 build requires a strongly typed FP16 ONNX input"
              : "TensorRT 11 FP32 build requires a strongly typed FP32 ONNX input");
    }

    InferPtr<nvinfer1::IBuilderConfig> config{
        builder->createBuilderConfig()};
    if (!config) throw std::runtime_error("createBuilderConfig failed");
    config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE,
        arguments.workspace_mib * 1024ULL * 1024ULL);
    std::cerr << "building TensorRT " << NV_TENSORRT_MAJOR << '.'
              << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH
              << " engine on " << device.name << " (sm_" << device.major
              << device.minor << ") precision="
              << (arguments.fp16 ? "fp16" : "fp32") << '\n';
    InferPtr<nvinfer1::IHostMemory> serialized{
        builder->buildSerializedNetwork(*network, *config)};
    if (!serialized) {
      throw std::runtime_error("buildSerializedNetwork returned null");
    }
    {
      std::ofstream stream(partial, std::ios::binary | std::ios::out);
      if (!stream) {
        throw std::runtime_error("cannot create partial engine: " +
                                 partial.string());
      }
      stream.write(static_cast<const char*>(serialized->data()),
                   static_cast<std::streamsize>(serialized->size()));
      if (!stream) throw std::runtime_error("engine write failed");
    }

    InferPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) throw std::runtime_error("createInferRuntime failed");
    InferPtr<nvinfer1::ICudaEngine> engine{runtime->deserializeCudaEngine(
        serialized->data(), serialized->size())};
    if (!engine) throw std::runtime_error("post-build deserialization failed");
    std::filesystem::rename(partial, output);

    std::cout << "engine=" << output << '\n'
              << "bytes=" << serialized->size() << '\n'
              << "tensorrt=" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR
              << '.' << NV_TENSORRT_PATCH << '\n'
              << "gpu=" << device.name << '\n'
              << "compute_capability=" << device.major << '.' << device.minor
              << '\n';
    for (int index = 0; index < engine->getNbIOTensors(); ++index) {
      const char* name = engine->getIOTensorName(index);
      std::cout << "io=" << name << ':'
                << (engine->getTensorIOMode(name) ==
                            nvinfer1::TensorIOMode::kINPUT
                        ? "input"
                        : "output")
                << ':' << dimensions(engine->getTensorShape(name)) << ':'
                << static_cast<int>(engine->getTensorDataType(name)) << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "autoaim_research_build_engine: " << error.what() << '\n';
    return 1;
  }
}

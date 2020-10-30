#include "resnet-mediapipe/include/hwc_to_chw_calculator.h"

#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"

/**
 * Converts a CPU tflite tensor that is in the format HWC to a CPU tflite tensor
 * in the format CHW
 **/
namespace mediapipe {
    constexpr char kInputTag[] = "IN_TENSOR";
    constexpr char kOutputTag[] = "OUT_TENSOR";

    REGISTER_CALCULATOR(HWCToCHWCalculator);

    Status HWCToCHWCalculator::GetContract(CalculatorContract* cc) {
        RET_CHECK(cc->Inputs().HasTag(kInputTag));
        RET_CHECK(cc->Outputs().HasTag(kOutputTag));

        cc->Inputs().Tag(kInputTag).Set<std::vector<TfLiteTensor>>();
        cc->Outputs().Tag(kOutputTag).Set<std::vector<TfLiteTensor>>();

        return OkStatus();
    }

    Status HWCToCHWCalculator::Open(CalculatorContext* cc) {
        cc->SetOffset(TimestampDiff(0));

        interpreter_ = absl::make_unique<tflite::Interpreter>();
        interpreter_->AddTensors(1);
        interpreter_->SetInputs({0});

        return OkStatus();
    }

    Status HWCToCHWCalculator::Process(CalculatorContext* cc) {
        RET_CHECK(!cc->Inputs().Tag(kInputTag).IsEmpty());

        // getting input tensors
        const std::vector<TfLiteTensor>& in_tensors = cc->Inputs().Tag(kInputTag).Get<std::vector<TfLiteTensor>>();

        // declaring variables
        auto output = absl::make_unique<std::vector<TfLiteTensor>>();

        // intializing interpreter if not initialized
        if (!initialized_) {
            TfLiteQuantization quant;
            quant.type = kTfLiteNoQuantization;
            quant.params = nullptr;
            interpreter_->SetTensorParametersReadWrite(
                /*tensor_index = */ 0,
                /*type = */ kTfLiteFloat32,
                /*name = */ "",
                /*dims = */ {3},
                /*quantization = */ quant
            );

            initialized_ = true;
        }

        for (int i = 0; i < in_tensors.size(); i++) {
            // getting input tensor and shapes
            const TfLiteTensor* t = &in_tensors[i];
            const tflite::RuntimeShape input_shape = tflite::GetTensorShape(t);
            const std::vector<int> output_shape = ApplyPermOnInputShape(input_shape, perms_);

            // getting result tensor from interpreter
            const int tensor_idx = interpreter_->inputs()[0];
            TfLiteTensor* res = interpreter_->tensor(tensor_idx);
            interpreter_->ResizeInputTensor(tensor_idx, output_shape);
            interpreter_->AllocateTensors();

            // transposing tensor
            Permute(tflite::GetTensorData<float>(t), input_shape, perms_, tflite::GetTensorData<float>(res));

            // adding resultant tensor to result vector
            output->emplace_back(*res);
        }

        cc->Outputs().Tag(kOutputTag).Add(output.release(), cc->InputTimestamp());
        return OkStatus();
    }

    Status HWCToCHWCalculator::Close(CalculatorContext* cc) {
        interpreter_.reset();
        return ::mediapipe::OkStatus();
    }

    void HWCToCHWCalculator::Permute(const float* in, const tflite::RuntimeShape& input_shape, const std::vector<int>& perms, float* out) {
        // creating output tensor shape
        tflite::RuntimeShape output_shape(perms.size());
        for (int i = 0; i < perms.size(); i++) {
            output_shape.SetDim(i, input_shape.Dims(perms[i]));
        }

        // creating transpose parameters
        tflite::TransposeParams params;
        params.perm_count = perms.size();
        for (int i = 0; i < perms.size(); ++i) {
            params.perm[i] = perms[i];
        }

        // transposing
        tflite::reference_ops::Transpose<float>(
            params,
            input_shape,
            in,
            output_shape,
            out
        );
    }

    std::vector<int> HWCToCHWCalculator::ApplyPermOnInputShape(const tflite::RuntimeShape& input_shape, const std::vector<int>& perms) {
        std::vector<int> output_shape;
        output_shape.resize(input_shape.DimensionsCount());

        for (int i = 0; i < perms.size(); i++) {
            output_shape[i] = input_shape.Dims(perms[i]);
        }

        return output_shape;
    }
}  // namespace mediapipe
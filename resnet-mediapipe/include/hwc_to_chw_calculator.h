#ifndef hwc_to_chw_calculator_H
#define hwc_to_chw_calculator_H

#include "mediapipe/framework/calculator_framework.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {
    class HWCToCHWCalculator : public CalculatorBase {
       public:
        static Status GetContract(CalculatorContract* cc);
        Status Open(CalculatorContext* cc);
        Status Process(CalculatorContext* cc);
        Status Close(CalculatorContext* cc);

       private:
        std::unique_ptr<tflite::Interpreter> interpreter_ = nullptr;
        bool initialized_ = false;
        const std::vector<int> perms_ = {2, 0, 1};
        
        void Permute(const float* in, const tflite::RuntimeShape& input_shape, const std::vector<int>& perms, float* out);
        std::vector<int> ApplyPermOnInputShape(const tflite::RuntimeShape& input_shape, const std::vector<int>& perms);
    };
}  // namespace mediapipe

#endif
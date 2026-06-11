#pragma once
#include "matrix.h"
namespace ncore {
    namespace nn {
        template <size_t InputDim, size_t HiddenDim, size_t OutputDim> class ContinuousMLP {
        public:
            math::MatrixF32<InputDim, HiddenDim> W1;
            math::MatrixF32<1, HiddenDim> B1;
            math::MatrixF32<HiddenDim, HiddenDim> W2;
            math::MatrixF32<1, HiddenDim> B2;
            math::MatrixF32<HiddenDim, OutputDim> W3;
            math::MatrixF32<1, OutputDim> B3;

            void load_weights(const float* weight_array) {
                const float* ptr = weight_array;

                for (size_t i = 0; i < InputDim * HiddenDim; i++) W1.data[i] = *ptr++;
                for (size_t i = 0; i < HiddenDim; i++)            B1.data[i] = *ptr++;

                for (size_t i = 0; i < HiddenDim * HiddenDim; i++) W2.data[i] = *ptr++;
                for (size_t i = 0; i < HiddenDim; i++)             B2.data[i] = *ptr++;

                for (size_t i = 0; i < HiddenDim * OutputDim; i++) W3.data[i] = *ptr++;
                for (size_t i = 0; i < OutputDim; i++)             B3.data[i] = *ptr++;
            }
            void forward(const float state[], const float control_inputs[], float out_derivatives[], size_t stateLength, size_t controlLength) const {
                math::MatrixF32<1, InputDim> in;
                math::MatrixF32<1, HiddenDim> l;
                math::MatrixF32<1, OutputDim> out;
                size_t index = 0;

                for (size_t i = 0; i < stateLength; i++) {
                    in.data[index++] = state[i];
                }
                for (size_t i = 0; i < controlLength; i++) {
                    in.data[index++] = control_inputs[i]; 
                }
                matmul(in, W1, l);
                add_bias(l, B1);
                apply_tanh(l);
                matmul(l, W2, l);
                add_bias(l, B2);
                apply_tanh(l);
                matmul(l, W3, out);
                add_bias(out, B3);
                //out_derivatives = out.data;
                for (float i : out.data) {
                    *out_derivatives = i;
                    out_derivatives++;
                }
            }            
        };
    }
}
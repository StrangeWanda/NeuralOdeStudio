// Pure C++ Matrix operations for MCU 
#include <tuple>
#include <math.h>
namespace ncore {
    // Sub-namespaces: math, nn, solver
    namespace math {
        template <size_t Rows, size_t Cols> struct MatrixF32 {
        public:
            float* data;
            MatrixF32(float d[])
            {
                float data[Rows * Cols] = {};
                for (size_t i = 0; i < Rows * Cols; i++) {
                    data[i] = d[i];
                }
            }
            MatrixF32() {
                float data[Rows * Cols];
            }
            ~MatrixF32() {
                delete[] data;
            }
            float& at(const size_t& i, const size_t& j) {
                return data[i * Cols + j];
            }
            [[nodiscard]] constexpr std::tuple<size_t, size_t> size() const {
                return { Rows, Cols };
            }
            float& operator()(const size_t& i, const size_t& j) {
                return this->at(i, j);
            }
        };
        template<size_t A, size_t B, size_t C>
        void matmul(const MatrixF32<A, B>& m1, const MatrixF32<B, C>& m2, MatrixF32<A, C>& out) {
            for (size_t i = 0; i < A; i++) {
                for (size_t j = 0; j < C; j++) {
                    float ans = 0;
                    for (size_t k = 0; k < B; k++) {
                        ans += m1.at(i, k) * m2.at(k, j);
                    }
                    out.at(i, j) = ans;
                }
            }
        }
        template <size_t Cols>
        void add_bias(MatrixF32<1, Cols>& layer_output, const MatrixF32<1, Cols>& bias) {
            for (size_t i = 0; i < Cols; i++) {
                layer_output.at(1, i) += bias(1, i);
            }
        }
        template <size_t Cols>
        void apply_relu(MatrixF32<1, Cols>& layer) {
            for (int i = 0; i < Cols; i++) {
                layer(1, i) = (layer(1, i) > 0.0f) ? layer(1, i) : 0.0f;
            }
        }
        template <size_t Cols>
        void apply_tanh(MatrixF32<1, Cols>& layer) {
            layer(1, Cols) = tanh(layer(1, Cols));
        }
    }
    namespace nn {
        template <size_t InputDim, size_t HiddenDim, size_t OutputDim> class ContinuousMLP {
        public:
            MatrixF32<InputDim, HiddenDim> W1;
            MatrixF32<1, HiddenDim> B1;
            MatrixF32<HiddenDim, HiddenDim> W2;
            MatrixF32<1, HiddenDim> B2;
            MatrixF32<HiddenDim, OutputDim> W3;
            MatrixF32<1, OutputDim> B3;

            void load_weights(const float* weight_array) {
                int i = 0;
                for (; i < InputDim * HiddenDim; i++) {
                    W1.data[i] = weight_array[i];
                }
                i++;
                for (; i < InputDim * HiddenDim + HiddenDim; i++) {
                    B1.data[i] = weight_array[i];
                }
                i++;
                for (; i < InputDim * HiddenDim + HiddenDim + HiddenDim * HiddenDim; i++) {
                    W2.data[i] = weight_array[i];
                }
                i++;
                for (; i < InputDim * HiddenDim + 2 * HiddenDim + HiddenDim * HiddenDim; i++) {
                    B2.data[i] = weight_array[i];
                }
                i++;
                for (; i < HiddenDim * OutputDim + InputDim * HiddenDim + 2 * HiddenDim + HiddenDim * HiddenDim; i++) {
                    W3.data[i] = weight_array[i];
                }
                i++;
                for (; i < HiddenDim * OutputDim + InputDim * HiddenDim + 2 * HiddenDim + HiddenDim * HiddenDim; i++) {
                    B3.data[i] = weight_array[i];
                }
                void forward(const float* state, const float* control_inputs, float* out_derivatives) const {
                    math::MatrixF32<1, InputDim> in;
                    math::MatrixF32<1, HiddenDim> l;
                    MatrixF32<1, OutputDim> out;
                    int index=0;
                    for (float s : state) {
                        in.data[index] = s;
                        index++;
                    }
                    for (float s : control_inputs) {
                        in.data[index] = s;
                        index++;
                    }
                    math::matmul(in, W1, l);
                    math::add_bias(l, B1);
                    math::matmul(l, W2, l);
                    math::add_bias(l, B2);
                    math::matmul(l, W3, out);
                    math::add_bias(out, B3);
                    return out.data;
                }
            }
        };
    }
    namespace solver {
        class RK4Solver {

        };
    }
}
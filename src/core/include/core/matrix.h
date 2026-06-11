// Pure C++ Matrix operations for MCU 
#include <tuple>
#include <math.h>
namespace ncore {
    // Sub-namespaces: math, nn, solver
    namespace math {
        template <size_t Rows, size_t Cols> struct MatrixF32 {
        public:
            float data[Rows * Cols] = {};
            MatrixF32(float d[])
            {
                for (size_t i = 0; i < Rows * Cols; i++) {
                    data[i] = d[i];
                }
            }
            MatrixF32() {}
            float& at(const size_t& i, const size_t& j) {
                return data[i * Cols + j];
            }
            float at(const size_t& i, const size_t& j) const {
                return data[i * Cols + j];
            }
            [[nodiscard]] constexpr std::tuple<size_t, size_t> size() const {
                return { Rows, Cols };
            }
            float& operator()(const size_t& i, const size_t& j) {
                return this->at(i, j);
            }
            float operator()(const size_t& i, const size_t& j) const {
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
                layer_output.at(0, i) += bias(0, i);
            }
        }
        template <size_t Cols>
        void apply_relu(MatrixF32<1, Cols>& layer) {
            for (int i = 0; i < Cols; i++) {
                layer(0, i) = (layer(0, i) > 0.0f) ? layer(0, i) : 0.0f;
            }
        }
        template <size_t Cols>
        void apply_tanh(MatrixF32<1, Cols>& layer) {
            for(int i=0;i<Cols;i++)
                layer(0, i) = tanh(layer(0, i));
        }
    }
}
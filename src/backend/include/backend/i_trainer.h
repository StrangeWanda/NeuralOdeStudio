#pragma once 

#include "core/neural_network.h"

namespace nbackend {

    template <size_t InputDim, size_t HiddenDim, size_t OutputDim, size_t StateDim, size_t ControlDim>
    class ITrainer {
    public:
        virtual ~ITrainer() = default;

        virtual void trainStep(
            ncore::nn::ContinuousMLP<InputDim, HiddenDim, OutputDim>& model,
            const float* initial_state,
            const float* control_inputs,
            const float* target_trajectory, 
            float t_end,
            float dt,
            float learning_rate
        ) = 0;

        
        static constexpr size_t getNumWeights() {
            return (InputDim * HiddenDim) + HiddenDim +
                (HiddenDim * HiddenDim) + HiddenDim +
                (HiddenDim * OutputDim) + OutputDim;
        }

        static constexpr size_t getAugmentedDim() {
            return StateDim + StateDim + get_num_weights();
        }
    };

}
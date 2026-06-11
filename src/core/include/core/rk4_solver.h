#pragma once
namespace ncore {
    namespace solver {
        template<size_t stateDim, size_t controlDim>
        class RK4Solver {
        public:
            float k1[stateDim];
            float k2[stateDim];
            float k3[stateDim];
            float k4[stateDim];
            float tempState[stateDim];
            template <typename Model> void step(const Model& mlp, float (&current_state)[stateDim], const float* control, float dt) {
                /*  Function to step forward int Runge-Kutta 4 via the Model. */
                mlp.forward(current_state, control, k1, stateDim, controlDim);
                // Move forward dt
                for (size_t i = 0; i < stateDim;i++) {
                    tempState[i] = current_state[i] + k1[i] * (dt / 2.0f);
                }
                mlp.forward(tempState, control, k2, stateDim, controlDim);
                for (size_t i = 0; i < stateDim; i++) {
                    tempState[i] = current_state[i] + k2[i] * (dt / 2.0f);
                }
                mlp.forward(tempState, control, k3, stateDim, controlDim);
                for (size_t i = 0; i < stateDim; i++) {
                    tempState[i] = current_state[i] + k3[i] * (dt);
                }
                mlp.forward(tempState, control, k4, stateDim, controlDim);
                for (size_t i = 0; i < stateDim; i++) {
                    current_state[i] = current_state[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * (dt / 6.0f);
                }
            }

            template <typename Model> void integrateTrajectory(const Model& mlp, float(&initial_state)[stateDim], const float* control, float t_end, float dt, float* trajectory_out) {
                /*A wrapper around step that runs a while loop from t=0 to t=t_end, saving the state at each step into trajectory_out so the GUI can chart it.*/
                float t = 0;
                //for (int i = 0; i < stateDim; i++) {
                //    trajectory_out[i] = initial_state[i];
                //}
                //trajectory_out += stateDim;
                int num_steps = static_cast<int>((t_end / dt) + 0.5f);
                for (int step_idx = 0; step_idx < num_steps; step_idx++) {
                    step(mlp, initial_state, control, dt);
                    t += dt;
                    for (int i = 0; i < stateDim; i++) {
                        trajectory_out[i] = initial_state[i];
                    }
                    trajectory_out += stateDim;
                }
            }
        };
    }
}
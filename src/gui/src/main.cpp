//

#include <iostream>
#include <iomanip>
#include "core/neural_network.h"
#include "core/rk4_solver.h"

int main() {
    std::cout << "--- Booting Neural ODE Core ---\n\n";

    // 1. Instantiate the Model (State: 2, Hidden: 4, Control: 2)
    ncore::nn::ContinuousMLP<4, 4, 2> model;

    // 2. Generate a dummy weight array (Total params for 2x4x2 = 34)
    float dummy_weights[34];
    for (int i = 0; i < 34; i++) {
        dummy_weights[i] = 0.1f; // Initialize everything to a small positive weight
    }
    model.load_weights(dummy_weights);
    std::cout << "[SUCCESS] Weights loaded safely without heap allocation.\n";

    // 3. Setup the Physics Parameters
    float initial_state[2] = { 0.0f, 0.0f };      // i_L = 0, v_C = 0
    float control_inputs[2] = { 0.5f, 24.0f };    // D = 0.5, V_in = 24V

    float dt = 1e-4f;  // 100 microsecond time step
    float t_end = 1e-3f; // 1 millisecond total simulation time
    int num_steps = static_cast<int>((t_end / dt) + 0.5f);

    // Create a stack buffer to hold the output trajectory (2 states * num_steps)
    float trajectory_buffer[20];

    // 4. Instantiate the Solver and Run
    ncore::solver::RK4Solver<2, 2> solver;
    std::cout << "[RUNNING] Integrating trajectory over " << num_steps << " steps...\n\n";

    solver.integrateTrajectory(model, initial_state, control_inputs, t_end, dt, trajectory_buffer);

    // 5. Print the Results
    std::cout << std::left << std::setw(10) << "Time(ms)"
        << std::setw(15) << "State 0 (i_L)"
        << std::setw(15) << "State 1 (v_C)" << "\n";
    std::cout << "------------------------------------------\n";

    for (int i = 0; i < num_steps; i++) {
        std::cout << std::left << std::setw(10) << (i * dt * 1000.0f) // Convert to ms
            << std::setw(15) << trajectory_buffer[i * 2]
            << std::setw(15) << trajectory_buffer[i * 2 + 1] << "\n";
    }

    std::cout << "\n--- Test Complete ---\n";
    return 0;
}
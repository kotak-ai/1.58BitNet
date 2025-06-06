# AGENT Instructions

This file outlines advanced research tasks derived from three papers. Implementations should address all tasks fully—no placeholders or pseudo-code. Refer to these lists when extending the repository.

## 1. Energentic Intelligence: From Self-Sustaining Systems to Enduring Artificial Life

### Tasks
- **Energy Utility Formulation**: Create an energy-based utility function for harvesting, consumption, and thermal load. Implement metrics Energetic Viability Score (EVS), Thermal Resilience Index (TRI), and Survival Horizon Expectation (SHE).
- **Learning Algorithm**: Implement a survival-focused RL algorithm (e.g., Q-learning variant) for balancing energy intake, computation, and thermal management.
- **Multi-Agent Considerations**: Implement policies for coordination and resource sharing among multiple agents.

### Tests
- Validate the energy budget under varying supply scenarios.
- Ensure thermal stability under different workloads.
- Evaluate survival horizon metrics over extended simulations.
- Observe behavioral transitions (dormant, active, degraded) due to energy/thermal changes.
- Run stress tests for extreme conditions.
- Test fairness when multiple agents compete for resources.

## 2. Multi-Layer GRPO: Enhancing Reasoning and Self-Correction in Large Language Models

### Tasks
- **Implement Base LLM and GRPO**: Select or build an LLM architecture and implement standard GRPO.
- **Design Multi-Layer Framework**: First layer generates a response via GRPO; the second layer reviews and corrects the output with another GRPO pass.
- **Self-Correction and Data Handling**: Discard failed corrections; reinforce successfully corrected samples with a correction-augmentation–selection mechanism.
- **Reward Structure**: Define reward signals for both layers, rewarding successful corrections.
- **Training Pipeline**: Prepare reasoning datasets and manage curriculum scheduling.
- **Benchmarking and Evaluation**: Use reasoning benchmarks to compare standard and multi-layer GRPO.
- **Scalability Considerations**: Optimize training efficiency to handle larger models and datasets.

### Tests
- Unit tests verifying both layers' outputs.
- Measure self-correction rate across a validation set.
- Compare reasoning accuracy before and after multi-layer training.
- Monitor reward stability during training.
- Conduct ablation studies removing the second layer or altering prompts.
- Record training time, memory use, and compute cost versus standard GRPO.

## 3. BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs

### Tasks
- **Model Architecture Preparation**: Modify a transformer to use BitLinear layers and integrate H-BitLinear modules.
- **Quantization Pipeline**: Implement 1-bit weight and 4-bit activation quantization with the Hadamard transform and dequantization where needed.
- **Training from Scratch vs. Conversion**: Provide options to train from scratch with 8-bit activations then fine-tune, or convert a pre-trained model.
- **Memory and Compute Optimization**: Evaluate memory/throughput and implement efficient kernels for packed formats.
- **Model Evaluation**: Measure performance on language modeling tasks, comparing to BitNet b1.58 and baselines.
- **Deployment Considerations**: Benchmark batched inference and ensure accuracy/latency trade-offs are acceptable.
- **Robustness and Compatibility**: Test across sequence lengths, batch sizes, and frameworks.

### Tests
- Verify quantization of weights and activations.
- Validate the Hadamard transform smoothing effect and reversibility.
- Monitor training convergence for 4-bit versus 8-bit activations.
- Benchmark inference speed and memory use.
- Evaluate downstream task performance to confirm minimal degradation.
- Stress test with long or out-of-distribution sequences.

### Integration Testing
- If combining methods from the three papers, interface the energy-aware controller with the quantized LLM and GRPO training. Ensure survival objectives remain compatible with reasoning tasks and on-device adaptation.


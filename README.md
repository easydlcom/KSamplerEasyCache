# KSamplerEasyCache

This repository provides a custom ComfyUI node, `KSampler (EasyCache)`, which integrates the **EasyCache** acceleration framework into the standard KSampler diffusion process. By dynamically detecting "stable periods" during video diffusion inference, EasyCache reuses historical computation results, significantly reducing redundant inference steps and accelerating video generation.

EasyCache is a training-free, model-agnostic, and plug-and-play acceleration framework.

## 🚀 Features

* **Accelerated Video Diffusion**: Integrates EasyCache's runtime-adaptive caching directly into the KSampler, speeding up video generation by reusing intermediate results during "stable periods" of the diffusion process.
* **Training-Free**: No model retraining or architecture modifications are required.
* **Seamless Integration**: Functions as a drop-in replacement for the standard KSampler node in your ComfyUI workflows.
* **Dynamic Caching**: Adapts to the diffusion process, ensuring full computation during critical early steps (warm-up) and leveraging caching in later, more stable phases.

## 📦 Installation

### 1. Clone the Repository

Navigate to your ComfyUI `custom_nodes` directory and clone this repository:

```bash
cd ComfyUI/custom_nodes
git clone [https://github.com/easydlcom/KSamplerEasyCache.git](https://github.com/easydlcom/KSamplerEasyCache.git)

```

### 2. Install EasyCache Dependencies

The KSamplerEasyCache node relies on the core EasyCache logic, which operates by modifying the underlying diffusion model's forward method. You need to ensure that the necessary dependencies for EasyCache (primarily related to the Wan2.1 model structure) are available in your ComfyUI Python environment.

## 🔌 Usage

### Launch ComfyUI

Add the Node: Right-click on the canvas, go to Add Node -> sampling -> custom_ksampler, and select KSampler (EasyCache).

Replace Standard KSampler: Use KSampler (EasyCache) as a direct replacement for your existing KSampler or KSamplerAdvanced nodes in your video generation workflows (e.g., Wan2.1 text-to-video or image-to-video workflows).

### Configure Parameters

All standard KSampler inputs (model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise) are available.

easycache_enabled: (Boolean, default: True) Enable or disable EasyCache acceleration. If disabled, it behaves like a standard KSampler.

easycache_threshold: (Float, default: 0.05) The accumulated error threshold (
tau). A smaller value means less caching but potentially higher quality; a larger value means more caching (greater acceleration) but potentially slightly lower quality. Experiment with values like 0.05 for 2.0x acceleration or 0.2 for 3.0x acceleration, as suggested by the EasyCache paper.

easycache_warmup_steps: (Integer, default: 10) The number of initial diffusion steps (R) during which EasyCache will always perform full computation. This ensures the model establishes the overall structure of the video before caching begins. (Note: Internally, this value is doubled to account for conditional and unconditional forward passes per step).

Example Workflow Snippet:
[Your Model Loader (e.g., Wan2.1 Model)] -> Model Input of KSampler (EasyCache)
[Your Positive Conditioning]               -> Positive Input of KSampler (EasyCache)
[Your Negative Conditioning]               -> Negative Input of KSampler (EasyCache)
[Your Latent Image / Empty Latent Image]   -> Latent Input of KSampler (EasyCache)

KSampler (EasyCache) Output -> [VAE Decode] -> [Save Image/Video]

## ⚠️ Important Notes

Wan2.1 Model Compatibility: This KSampler (EasyCache) node is specifically designed to work with diffusion models whose forward method signature matches that expected by the EasyCache framework (e.g., Wan2.1 models). Ensure your ComfyUI environment can correctly load and use Wan2.1 models.

First Run vs. Subsequent Runs: EasyCache's acceleration is most noticeable on subsequent runs with the same or highly similar inputs, or during internal iterations of video generation where frames are highly correlated. The very first run will populate the cache.

Cleanup: The node automatically cleans up the EasyCache internal state and restores the original model's forward method after sampling is complete, preventing side effects.

Debugging: If you encounter issues, check your ComfyUI console for EasyCache-related print statements that indicate its internal state and caching decisions.

## 📄 License

This project is open-sourced under the Apache-2.0 License.

## ❤️ Acknowledgements

EasyCache Project: We extend our gratitude to the authors of the original EasyCache project for their innovative work on diffusion model acceleration.

ComfyUI: Thanks to the ComfyUI developers for creating such a flexible and powerful node-based UI for Stable Diffusion.
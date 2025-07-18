import torch
import os
import folder_paths
import comfy.model_management as model_management
import comfy.samplers as samplers
import comfy.sd
from tqdm import tqdm
import torch.nn.functional as F
import math
import numpy as np
import copy
import hashlib # For robust cache key generation

# --- EasyCacheModelWrapper Class ---
# This class wraps the original model and contains the caching logic and state.
class EasyCacheModelWrapper(torch.nn.Module):
    def __init__(self, inner_model, easycache_threshold, easycache_warmup_steps, total_steps):
        super().__init__()
        self.inner_model = inner_model
        
        # EasyCache state variables, now instance-specific
        self.easycache_cnt = 0
        self.easycache_thresh = easycache_threshold
        self.easycache_ret_steps = easycache_warmup_steps * 2 # Multiply by 2 for cond/uncond passes
        self.easycache_cutoff_steps = total_steps * 2 - 2 # Total forward passes - 2 (for last pair)
        self.easycache_accumulated_error_even = 0
        self.easycache_should_calc_current_pair = True
        self.easycache_previous_raw_input_even = None
        self.easycache_previous_raw_output_even = None
        self.easycache_previous_raw_output_odd = None
        self.easycache_prev_prev_raw_input_even = None
        self.easycache_cache_even = None
        self.easycache_cache_odd = None
        self.easycache_num_steps = total_steps * 2
        self.easycache_k = None # To detect new sampling runs

        # Cache storage (simple dict for now, can be extended with LRU)
        self.cache = {}

    # Placeholder for more complex key generation (not strictly used in current EasyCache logic, but good practice)
    def _generate_cache_key(self, x, timesteps, context):
        # A simple key based on statistical properties and timestep
        # For actual EasyCache, it relies on sequential calls and internal state.
        # This function would be more relevant if we were caching arbitrary 'forward' calls.
        
        # Hash of timestep
        t_hash = str(timesteps.item()) if isinstance(timesteps, torch.Tensor) else str(timesteps)
        
        # For x, use a small, fixed-size representation (e.g., mean and std deviation)
        # Avoid full tensor hashing for performance
        x_mean = x.mean().item()
        x_std = x.std().item()
        
        # For context (conditioning), hashing its content can be complex.
        # For simplicity, if context is a list of tensors/dicts, hash a summary.
        # For now, relying on sequential nature of EasyCache.
        
        # This key isn't directly used by EasyCache's logic, which relies on easycache_cnt and state.
        # It's a conceptual suggestion for a more general caching wrapper.
        return f"{t_hash}_{x_mean:.4f}_{x_std:.4f}"

    def forward(self, x, timesteps, context, **kwargs):
        """
        Intercepts the model's forward call and applies EasyCache logic.
        """
        # Map ComfyUI's 'timesteps' to EasyCache/Wan's 't'
        t = timesteps

        # Determine if it's an even or odd step for caching logic
        is_even = self.easycache_cnt % 2 == 0

        if self.easycache_cnt < self.easycache_ret_steps or self.easycache_should_calc_current_pair:
            # Perform full calculation (warmup phase or when error threshold is exceeded)
            output = self.inner_model(x, timesteps, context, **kwargs)

            if is_even:
                self.easycache_previous_raw_input_even = x.detach().clone()
                self.easycache_previous_raw_output_even = output.detach().clone()
                self.easycache_cache_even = output.detach().clone() # Cache for even steps
            else:
                self.easycache_previous_raw_output_odd = output.detach().clone()
                self.easycache_cache_odd = output.detach().clone() # Cache for odd steps

            self.easycache_should_calc_current_pair = False # Reset for next pair
            self.easycache_accumulated_error_even = 0 # Reset error after full calculation
        else:
            # Try to use cached results
            if is_even:
                if self.easycache_cache_even is not None:
                    output = self.easycache_cache_even.clone()
                else:
                    # Fallback if cache is somehow empty (should not happen in normal flow)
                    print("EasyCache: Even cache miss, performing full computation.")
                    output = self.inner_model(x, timesteps, context, **kwargs)
            else: # Odd step
                if self.easycache_cache_odd is not None:
                    output = self.easycache_cache_odd.clone()
                else:
                    print("EasyCache: Odd cache miss, performing full computation.")
                    output = self.inner_model(x, timesteps, context, **kwargs)

            # Update accumulated error for even steps (this logic is specific to EasyCache)
            if is_even and self.easycache_previous_raw_input_even is not None and self.easycache_cnt > 0:
                if self.easycache_cnt == self.easycache_ret_steps:
                    self.easycache_prev_prev_raw_input_even = self.easycache_previous_raw_input_even.detach().clone()
                else:
                    if self.easycache_prev_prev_raw_input_even is not None:
                        # Calculate L1 distance (mean absolute error) between current and previous input
                        l1_distance = torch.mean(torch.abs(x - self.easycache_prev_prev_raw_input_even))
                        self.easycache_accumulated_error_even += l1_distance.item()

                if self.easycache_accumulated_error_even > self.easycache_thresh:
                    self.easycache_should_calc_current_pair = True
                    self.easycache_accumulated_error_even = 0 # Reset error after triggering recalculation

                self.easycache_prev_prev_raw_input_even = self.easycache_previous_raw_input_even.detach().clone()
                self.easycache_previous_raw_input_even = x.detach().clone()
                self.easycache_previous_raw_output_even = output.detach().clone()

        self.easycache_cnt += 1 # Increment step counter

        return output

# --- KSampler (EasyCache) Custom Node ---
class KSamplerEasyCache:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.filename_prefix = 'ComfyUI'

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (samplers.KSampler.SAMPLERS,),
                "scheduler": (samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "easycache_enabled": ("BOOLEAN", {"default": True}),
                "easycache_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "easycache_warmup_steps": ("INT", {"default": 10, "min": 1, "max": 50}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_ksampler"
    OUTPUT_NODE = True # Indicates this node has direct outputs

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
               easycache_enabled, easycache_threshold, easycache_warmup_steps):

        if not easycache_enabled:
            print("EasyCache is disabled. Running standard KSampler.")
            return samplers.sample(
                model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
            )

        # Store the original underlying model from the ModelPatcher
        original_underlying_model = model.model
        
        print(f"EasyCache enabled: threshold={easycache_threshold}, warmup_steps={easycache_warmup_steps}")

        try:
            # Debugging prints (can be commented out for production use)
            print(f"DEBUG KSamplerEasyCache Input: Type of 'positive': {type(positive)}")
            print(f"DEBUG KSamplerEasyCache Input: First element of 'positive': {str(positive[0])[:100]}...")
            print(f"DEBUG KSamplerEasyCache Input: Type of 'negative': {type(negative)}")
            print(f"DEBUG KSamplerEasyCache Input: First element of 'negative': {str(negative[0])[:100]}...")

            # Perform deepcopy on positive and negative conditioning lists
            # This helps to isolate them from any potential unexpected in-place modifications
            positive_copy = copy.deepcopy(positive)
            negative_copy = copy.deepcopy(negative)

            # Create an instance of our wrapper model
            # This wrapper will manage its own EasyCache state and call the original model's forward.
            easycache_wrapper_instance = EasyCacheModelWrapper(
                inner_model=original_underlying_model,
                easycache_threshold=easycache_threshold,
                easycache_warmup_steps=easycache_warmup_steps,
                total_steps=steps
            )

            # Temporarily replace the underlying model in the ComfyUI ModelPatcher
            # Now, when ComfyUI's samplers call model.forward(), it will be routed to our wrapper.
            model.model = easycache_wrapper_instance

            # Call ComfyUI's standard samplers.sample
            latent_output = samplers.sample(
                model, seed, steps, cfg, sampler_name, scheduler, positive_copy, negative_copy, latent_image, denoise
            )

            print("EasyCache KSampler sampling completed.")

        finally:
            # IMPORTANT: Always restore the original underlying model to the ModelPatcher
            # This ensures that other nodes or subsequent runs use the unpatched model.
            model.model = original_underlying_model

        return (latent_output,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "KSampler (EasyCache)": KSamplerEasyCache
}

# A dictionary that contains the friendly names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "KSampler (EasyCache)": "KSampler (EasyCache)"
}
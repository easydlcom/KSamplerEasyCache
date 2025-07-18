import torch
import os
# Add the missing import statement for folder_paths
import folder_paths # <--- ADD THIS LINE
import comfy.model_management as model_management
import comfy.samplers as samplers
import comfy.sd
from tqdm import tqdm
import torch.nn.functional as F
import math
import numpy as np

# --- EasyCache Core Logic Wrapper ---
# This function intercepts the model.forward calls and applies EasyCache logic.
def easycache_forward_wrapper(original_forward_func, instance, x, timesteps, context, **kwargs):
    """
    Wraps the original model's forward method to integrate EasyCache acceleration.
    Receives arguments as ComfyUI's model.forward usually passes them.
    """
    # Map ComfyUI's 'timesteps' to EasyCache/Wan's 't'
    t = timesteps

    # Extract other potential Wan-specific arguments from kwargs.
    # If ComfyUI's model.forward doesn't provide these, they will remain None.
    seq_len = kwargs.get('seq_len')
    clip_fea = kwargs.get('clip_fea')
    y = kwargs.get('y')

    # Ensure this is the actual model module being patched, not a wrapper.
    # The 'instance' object now holds the easycache_ state variables.

    # === EasyCache State Initialization and Logic ===
    # Check if a new sampling process or a new batch has started.
    # This logic assumes sequential calls to the forward function during sampling.
    if instance.easycache_k is None: # First call for a new KSampler run or new batch
        instance.easycache_k = torch.zeros(instance.easycache_num_steps, dtype=torch.long, device=x.device)
        instance.easycache_cnt = 0
        instance.easycache_accumulated_error_even = 0
        instance.easycache_should_calc_current_pair = True
        instance.easycache_previous_raw_input_even = None
        instance.easycache_previous_raw_output_even = None
        instance.easycache_previous_raw_output_odd = None
        instance.easycache_prev_prev_raw_input_even = None
        instance.easycache_cache_even = None
        instance.easycache_cache_odd = None


    # Determine if it's an even or odd step for caching logic
    is_even = instance.easycache_cnt % 2 == 0

    if instance.easycache_cnt < instance.easycache_ret_steps or instance.easycache_should_calc_current_pair:
        # Perform full calculation (warmup phase or when error threshold is exceeded)
        # Call the original forward function with the arguments it expects from ComfyUI
        output = original_forward_func(x, timesteps, context, **kwargs)

        if is_even:
            instance.easycache_previous_raw_input_even = x.detach().clone()
            instance.easycache_previous_raw_output_even = output.detach().clone()
            instance.easycache_cache_even = output.detach().clone() # Cache for even steps
        else:
            instance.easycache_previous_raw_output_odd = output.detach().clone()
            instance.easycache_cache_odd = output.detach().clone() # Cache for odd steps

        instance.easycache_should_calc_current_pair = False # Reset for next pair
        instance.easycache_accumulated_error_even = 0 # Reset error after full calculation
    else:
        # Try to use cached results
        if is_even:
            if instance.easycache_cache_even is not None:
                output = instance.easycache_cache_even.clone()
            else:
                # Fallback if cache is somehow empty (should not happen in normal flow)
                print("EasyCache: Even cache miss, performing full computation.")
                output = original_forward_func(x, timesteps, context, **kwargs)
        else: # Odd step
            if instance.easycache_cache_odd is not None:
                output = instance.easycache_cache_odd.clone()
            else:
                print("EasyCache: Odd cache miss, performing full computation.")
                output = original_forward_func(x, timesteps, context, **kwargs)

        # Update accumulated error for even steps (this logic is specific to EasyCache)
        if is_even and instance.easycache_previous_raw_input_even is not None and instance.easycache_cnt > 0:
            if instance.easycache_cnt == instance.easycache_ret_steps:
                 instance.easycache_prev_prev_raw_input_even = instance.easycache_previous_raw_input_even.detach().clone()
            else:
                if instance.easycache_prev_prev_raw_input_even is not None:
                    # Calculate L1 distance (mean absolute error) between current and previous input
                    l1_distance = torch.mean(torch.abs(x - instance.easycache_prev_prev_raw_input_even))
                    instance.easycache_accumulated_error_even += l1_distance.item()

            if instance.easycache_accumulated_error_even > instance.easycache_thresh:
                instance.easycache_should_calc_current_pair = True
                instance.easycache_accumulated_error_even = 0 # Reset error after triggering recalculation

            instance.easycache_prev_prev_raw_input_even = instance.easycache_previous_raw_input_even.detach().clone()
            instance.easycache_previous_raw_input_even = x.detach().clone()
            instance.easycache_previous_raw_output_even = output.detach().clone()

    instance.easycache_cnt += 1 # Increment step counter

    return output

# --- KSampler (EasyCache) Custom Node ---
class KSamplerEasyCache:
    def __init__(self):
        # Correctly access get_output_directory via the imported folder_paths module
        self.output_dir = folder_paths.get_output_directory() # <--- MODIFIED LINE
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

        # Get the actual torch.nn.Module model from the ComfyUI wrapper
        actual_model_module = model.model
        original_forward = actual_model_module.forward

        # Prepare EasyCache's state variables and inject them into the actual model module.
        # These are instance-specific and will be cleaned up in the 'finally' block.
        actual_model_module.easycache_cnt = 0
        actual_model_module.easycache_thresh = easycache_threshold
        # EasyCache internally counts conditional and unconditional passes, so multiply steps by 2
        actual_model_module.easycache_ret_steps = easycache_warmup_steps * 2
        actual_model_module.easycache_cutoff_steps = steps * 2 - 2 # Total forward passes - 2 (for last pair)
        actual_model_module.easycache_accumulated_error_even = 0
        actual_model_module.easycache_should_calc_current_pair = True
        actual_model_module.easycache_previous_raw_input_even = None
        actual_model_module.easycache_previous_raw_output_even = None
        actual_model_module.easycache_previous_raw_output_odd = None
        actual_model_module.easycache_prev_prev_raw_input_even = None
        actual_model_module.easycache_cache_even = None
        actual_model_module.easycache_cache_odd = None
        actual_model_module.easycache_num_steps = steps * 2
        actual_model_module.easycache_k = None # Initialize to None to detect new sampling runs

        # Define the patched forward function.
        # This signature (x, timesteps, context, **kwargs) matches how ComfyUI
        # typically calls the underlying model's forward method.
        def _patched_forward(x, timesteps, context, **kwargs):
            # Pass all arguments directly to our easycache_forward_wrapper,
            # along with the original forward function and the model instance.
            return easycache_forward_wrapper(original_forward, actual_model_module, x, timesteps, context, **kwargs)

        # Apply the patch: replace the model's original forward with our patched version.
        actual_model_module.forward = _patched_forward

        print(f"EasyCache enabled: threshold={easycache_threshold}, warmup_steps={easycache_warmup_steps}")

        try:
            # Debugging prints (can be commented out for production use)
            print(f"DEBUG KSamplerEasyCache Input: Type of 'positive': {type(positive)}")
            print(f"DEBUG KSamplerEasyCache Input: Type of 'negative': {type(negative)}")
            if isinstance(positive, (list, tuple)) and len(positive) > 0:
                print(f"DEBUG KSamplerEasyCache Input: First element of 'positive': {str(positive[0])[:100]}...")
            if isinstance(negative, (list, tuple)) and len(negative) > 0:
                print(f"DEBUG KSamplerEasyCache Input: First element of 'negative': {str(negative[0])[:100]}...")

            # Call ComfyUI's standard samplers.sample, which will now use our patched model.forward
            latent_output = samplers.sample(
                model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
            )

            print("EasyCache KSampler sampling completed.")

        finally:
            # IMPORTANT: Always restore the original forward method to prevent side effects
            # for subsequent operations or other nodes in the workflow.
            actual_model_module.forward = original_forward

            # Clean up EasyCache specific attributes from the model instance
            # to ensure a clean state for future runs or other model uses.
            if hasattr(actual_model_module, 'easycache_cnt'): # Check before deleting
                del actual_model_module.easycache_cnt
            if hasattr(actual_model_module, 'easycache_thresh'):
                del actual_model_module.easycache_thresh
            if hasattr(actual_model_module, 'easycache_ret_steps'):
                del actual_model_module.easycache_ret_steps
            if hasattr(actual_model_module, 'easycache_cutoff_steps'):
                del actual_model_module.easycache_cutoff_steps
            if hasattr(actual_model_module, 'easycache_accumulated_error_even'):
                del actual_model_module.easycache_accumulated_error_even
            if hasattr(actual_model_module, 'easycache_should_calc_current_pair'):
                del actual_model_module.easycache_should_calc_current_pair
            if hasattr(actual_model_module, 'easycache_previous_raw_input_even'):
                del actual_model_module.easycache_previous_raw_input_even
            if hasattr(actual_model_module, 'easycache_previous_raw_output_even'):
                del actual_model_module.easycache_previous_raw_output_even
            if hasattr(actual_model_module, 'easycache_previous_raw_output_odd'):
                del actual_model_module.easycache_previous_raw_output_odd
            if hasattr(actual_model_module, 'easycache_prev_prev_raw_input_even'):
                del actual_model_module.easycache_prev_prev_raw_input_even
            if hasattr(actual_model_module, 'easycache_cache_even'):
                del actual_model_module.easycache_cache_even
            if hasattr(actual_model_module, 'easycache_cache_odd'):
                del actual_model_module.easycache_cache_odd
            if hasattr(actual_model_module, 'easycache_num_steps'):
                del actual_model_module.easycache_num_steps
            if hasattr(actual_model_module, 'easycache_k'):
                del actual_model_module.easycache_k

        return (latent_output,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "KSampler (EasyCache)": KSamplerEasyCache
}

# A dictionary that contains the friendly names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "KSampler (EasyCache)": "KSampler (EasyCache)"
}
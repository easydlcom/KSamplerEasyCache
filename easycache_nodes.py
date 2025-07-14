import torch
import os
import comfy.model_management as model_management
import comfy.samplers as samplers
import comfy.sd as sd
from tqdm import tqdm # 用于显示进度条

# 定义 EasyCache 的缓存目录，可以放在 ComfyUI 的 temp 或 output 目录下
# 这里为了方便插件管理，我们仍然放在插件自己的目录下
EASYCACHE_DATA_DIR = os.path.join(os.path.dirname(__file__), "easycache_data")
os.makedirs(EASYCACHE_DATA_DIR, exist_ok=True)

# EasyCache 的核心逻辑，直接从 easycache_generate.py 复制并修改
def easycache_forward_wrapper(original_forward_func, instance, x, t, context, seq_len, clip_fea=None, y=None):
    """
    这是一个包装函数，用于替换模型的原始 forward 方法，并注入 EasyCache 逻辑。
    `instance` 是模型对象本身（例如 Wan 模型实例），EasyCache 的状态会存储在其属性中。
    """
    
    # 确保 EasyCache 状态变量已初始化
    # 这些变量在 KSamplerEasyCache 节点的 sample 方法中会被初始化到模型实例上
    if not hasattr(instance, 'easycache_cnt'): instance.easycache_cnt = 0
    if not hasattr(instance, 'easycache_thresh'): instance.easycache_thresh = 0.05
    if not hasattr(instance, 'easycache_ret_steps'): instance.easycache_ret_steps = 20 # 10 * 2
    if not hasattr(instance, 'easycache_cutoff_steps'): instance.easycache_cutoff_steps = 0 # 会在Ksampler中计算
    if not hasattr(instance, 'easycache_accumulated_error_even'): instance.easycache_accumulated_error_even = 0
    if not hasattr(instance, 'easycache_should_calc_current_pair'): instance.easycache_should_calc_current_pair = True
    if not hasattr(instance, 'easycache_k'): instance.easycache_k = None
    if not hasattr(instance, 'easycache_previous_raw_input_even'): instance.easycache_previous_raw_input_even = None
    if not hasattr(instance, 'easycache_previous_raw_output_even'): instance.easycache_previous_raw_output_even = None
    if not hasattr(instance, 'easycache_previous_raw_output_odd'): instance.easycache_previous_raw_output_odd = None
    if not hasattr(instance, 'easycache_prev_prev_raw_input_even'): instance.easycache_prev_prev_raw_input_even = None
    if not hasattr(instance, 'easycache_cache_even'): instance.easycache_cache_even = None
    if not hasattr(instance, 'easycache_cache_odd'): instance.easycache_cache_odd = None
    if not hasattr(instance, 'easycache_num_steps'): instance.easycache_num_steps = 0 # 总步数


    # Store original raw input for end-to-end caching
    raw_input = [u.clone() for u in x]

    # Track which type of step (even=condition, odd=uncondition)
    # self.cnt 对应 instance.easycache_cnt
    is_even = (instance.easycache_cnt % 2 == 0)

    # Only make decision on even (condition) steps
    if is_even:
        # Always compute first ret_steps and last steps
        if instance.easycache_cnt < instance.easycache_ret_steps or instance.easycache_cnt >= instance.easycache_cutoff_steps:
            instance.easycache_should_calc_current_pair = True
            instance.easycache_accumulated_error_even = 0
        else:
            # Check if we have previous step data for comparison
            if instance.easycache_previous_raw_input_even is not None and instance.easycache_previous_raw_output_even is not None:
                # Calculate input changes
                raw_input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(raw_input, instance.easycache_previous_raw_input_even)
                ]).abs().mean()

                # Compute predicted change if we have k factors
                if instance.easycache_k is not None:
                    # Calculate output norm for relative comparison
                    output_norm = torch.cat([
                        u.flatten() for u in instance.easycache_previous_raw_output_even
                    ]).abs().mean()
                    pred_change = instance.easycache_k * (raw_input_change / output_norm)
                    combined_pred_change = pred_change # easycache_generate.py 里面这里是直接赋值
                    # Accumulate predicted error
                    instance.easycache_accumulated_error_even += combined_pred_change
                    # Decide if we need full calculation
                    if instance.easycache_accumulated_error_even < instance.easycache_thresh:
                        instance.easycache_should_calc_current_pair = False
                    else:
                        instance.easycache_should_calc_current_pair = True
                        instance.easycache_accumulated_error_even = 0
                else:
                    # First time after ret_steps or missing k factors, need to calculate
                    instance.easycache_should_calc_current_pair = True
            else:
                # No previous data yet, must calculate
                instance.easycache_should_calc_current_pair = True

        # Store current input state
        instance.easycache_previous_raw_input_even = [u.clone().detach() for u in raw_input] # 使用 detach() 避免在缓存时创建计算图

    # Check if we can use cached output and return early
    if is_even and not instance.easycache_should_calc_current_pair and \
            instance.easycache_previous_raw_output_even is not None:
        # Use cached output directly
        instance.easycache_cnt += 1
        # Check if we've reached the end of sampling
        if instance.easycache_cnt >= instance.easycache_num_steps:
            instance.easycache_cnt = 0
            # easycache_generate.py 中这里还会清空 skip_cond_step/skip_uncond_step，但在我们这里不适用
            # 因为每次 KSampler 都重新设置了 EasyCache 状态，所以不需要手动清空
        
        # 确保返回的 Tensor 在正确的设备上
        # 原始 easycache_forward 返回 [(u + v).float() for u, v in zip(raw_input, self.cache_even)]
        # 假设 cache_even 已经包含了噪声预测，直接加到 raw_input 上
        # 这里需要特别注意，EasyCache 缓存的是 `output - raw_input`
        # 所以返回时需要 `raw_input + cache`
        # 并且要保证数据类型匹配，并且在正确的设备上
        
        cached_output = [(u + v).to(x[0].device).float() for u, v in zip(raw_input, instance.easycache_cache_even)]
        
        # print(f"EasyCache: Cached (even) - step {instance.easycache_cnt - 1}") # 打印信息方便调试
        return cached_output

    elif not is_even and not instance.easycache_should_calc_current_pair and \
            instance.easycache_previous_raw_output_odd is not None:
        # Use cached output directly
        instance.easycache_cnt += 1
        # Check if we've reached the end of sampling
        if instance.easycache_cnt >= instance.easycache_num_steps:
            instance.easycache_cnt = 0
        
        # 同样，返回时需要 `raw_input + cache_odd`
        cached_output = [(u + v).to(x[0].device).float() for u, v in zip(raw_input, instance.easycache_cache_odd)]

        # print(f"EasyCache: Cached (odd) - step {instance.easycache_cnt - 1}") # 打印信息方便调试
        return cached_output

    # Continue with normal processing since we need to calculate
    # 调用模型的原始 forward 方法
    # print(f"EasyCache: Full calculation ({'even' if is_even else 'odd'}) - step {instance.easycache_cnt}") # 打印信息方便调试
    output = original_forward_func(x, t, context, seq_len, clip_fea=clip_fea, y=y)

    # Update cache and calculate change rates if needed
    if is_even:  # Condition path
        # If we have previous output, calculate k factors for future predictions
        if instance.easycache_previous_raw_output_even is not None:
            # Calculate output change at the raw level
            output_change = torch.cat([
                (u - v).flatten() for u, v in zip(output, instance.easycache_previous_raw_output_even)
            ]).abs().mean()

            # Check if we have previous input state for comparison
            if instance.easycache_prev_prev_raw_input_even is not None:
                # Calculate input change
                input_change = torch.cat([
                    (u - v).flatten() for u, v in zip(
                        instance.easycache_previous_raw_input_even, instance.easycache_prev_prev_raw_input_even
                    )
                ]).abs().mean()
                
                # 避免除以零
                if input_change != 0:
                    instance.easycache_k = output_change / input_change
                else:
                    instance.easycache_k = None # 如果输入没有变化，k 无意义，设为 None

        instance.easycache_prev_prev_raw_input_even = instance.easycache_previous_raw_input_even # 复制前一个输入
        instance.easycache_previous_raw_output_even = [u.clone().detach() for u in output]
        instance.easycache_cache_even = [u - v for u, v in zip(output, raw_input)] # 缓存噪声残差

    else:  # Uncondition path
        # Store output for unconditional path
        instance.easycache_previous_raw_output_odd = [u.clone().detach() for u in output]
        instance.easycache_cache_odd = [u - v for u, v in zip(output, raw_input)] # 缓存噪声残差

    # Update counter
    instance.easycache_cnt += 1
    # 每次 KSampler 运行时，EasyCache 的状态会被重置，所以这里不需要清空 skip_cond_step/skip_uncond_step
    # 并且 easycache_cnt 的重置也由 KSamplerEasyCache 节点控制

    return [u.float() for u in output]


class KSamplerEasyCache:
    CATEGORY = "sampling/custom_ksampler" # 放在一个自定义的 KSampler 分类下

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
                # EasyCache 特定参数
                "easycache_enabled": ("BOOLEAN", {"default": True}),
                "easycache_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "easycache_warmup_steps": ("INT", {"default": 10, "min": 0, "max": 100}), # 对应论文中的 R
                # easycache_generate.py 中没有直接的 cache_key 参数，
                # 但它的状态是挂载在模型实例上的，所以每次运行都是独立的。
                # 如果未来需要区分不同视频/提示词的缓存，可能需要更复杂的管理
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    OUTPUT_NODE = True

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
               easycache_enabled, easycache_threshold, easycache_warmup_steps):
        
        if not easycache_enabled:
            # 如果 EasyCache 未启用，直接调用原始 KSampler 逻辑
            print("EasyCache is disabled. Running standard KSampler.")
            return samplers.sample(
                model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
            )

        # 获取 ComfyUI MODEL 包装器中的实际 torch.nn.Module
        # ComfyUI 的 MODEL 对象内部包含一个 model 属性，它才是实际的 nn.Module
        # 确保这个 model.model 是扩散模型 (如 UNet)
        actual_model_module = model.model # 这是一个 torch.nn.Module 实例

        # 保存原始的 forward 方法
        original_forward = actual_model_module.forward
        
        # 准备 EasyCache 的状态变量并注入到 actual_model_module 实例上
        # 这样 easycache_forward_wrapper 可以通过 `instance` (即 actual_model_module) 访问这些状态
        actual_model_module.easycache_cnt = 0
        actual_model_module.easycache_thresh = easycache_threshold
        # easycache_warmup_steps 对应论文中的 R，但在 easycache_generate.py 中它是 `ret_steps * 2`
        # 因为 `cnt` 每次递增 1，而实际扩散步是 `cnt // 2`
        actual_model_module.easycache_ret_steps = easycache_warmup_steps * 2 
        # cutoff_steps 对应 `sample_steps * 2 - 2`
        actual_model_module.easycache_cutoff_steps = steps * 2 - 2 
        actual_model_module.easycache_accumulated_error_even = 0
        actual_model_module.easycache_should_calc_current_pair = True
        actual_model_module.easycache_k = None
        actual_model_module.easycache_previous_raw_input_even = None
        actual_model_module.easycache_previous_raw_output_even = None
        actual_model_module.easycache_previous_raw_output_odd = None
        actual_model_module.easycache_prev_prev_raw_input_even = None
        actual_model_module.easycache_cache_even = None
        actual_model_module.easycache_cache_odd = None
        actual_model_module.easycache_num_steps = steps * 2 # 总的 forward 调用次数 (steps * 2)

        # 替换模型的 forward 方法
        # 我们需要创建一个 lambda 函数或 wrapper 来传递原始 forward 方法
        # easycache_forward_wrapper 会作为新的 forward 方法被调用
        # 它的第一个参数将是 `self` (即 actual_model_module)，
        # 然后是原始的 `forward` 参数 (x, t, context, seq_len, ...)
        def _patched_forward(*args, **kwargs):
            return easycache_forward_wrapper(original_forward, actual_model_module, *args, **kwargs)

        actual_model_module.forward = _patched_forward
        
        print(f"EasyCache enabled: threshold={easycache_threshold}, warmup_steps={easycache_warmup_steps}")

        try:
            # 调用 ComfyUI 内部的采样逻辑
            # EasyCache 的加速发生在 model.model.forward 内部，对 ComfyUI KSampler 来说是透明的
            latent_output = samplers.sample(
                model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
            )
            
            print("EasyCache KSampler sampling completed.")
            
        finally:
            # 无论成功或失败，都恢复原始的 forward 方法，避免副作用
            actual_model_module.forward = original_forward
            # 清理 EasyCache 相关的属性，防止在模型被重用时出现问题
            del actual_model_module.easycache_cnt
            del actual_model_module.easycache_thresh
            del actual_model_module.easycache_ret_steps
            del actual_model_module.easycache_cutoff_steps
            del actual_model_module.easycache_accumulated_error_even
            del actual_model_module.easycache_should_calc_current_pair
            del actual_model_module.easycache_k
            del actual_model_module.easycache_previous_raw_input_even
            del actual_model_module.easycache_previous_raw_output_even
            del actual_model_module.easycache_previous_raw_output_odd
            del actual_model_module.easycache_prev_prev_raw_input_even
            del actual_model_module.easycache_cache_even
            del actual_model_module.easycache_cache_odd
            del actual_model_module.easycache_num_steps
            
            # 强制清空 CUDA 缓存，虽然不是严格必要，但有助于释放一些内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        return (latent_output,)


# 节点映射表
NODE_CLASS_MAPPINGS = {
    "KSamplerEasyCache": KSamplerEasyCache,
}

# 节点显示名称映射表
NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerEasyCache": "KSampler (EasyCache)",
}
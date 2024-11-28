# https://gist.github.com/laksjdjf/435c512bc19636e9c9af4ee7bea9eb86
'''
https://arxiv.org/abs/2312.00858
1. put this file in ComfyUI/custom_nodes
2. load node from <loaders>
start_step, end_step: apply this method when the timestep is between start_step and end_step
cache_interval: interval of caching (1 means no caching)
cache_depth: depth of caching
'''

import torch
from torch import nn
from comfy.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed, timestep_embedding, th, apply_control

class DeepCacheV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "cache_interval": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "cache_depth": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 12,
                    "step": 1,
                    "display": "number"
                }),
                "start_step": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "end_step": ("INT", {
                    "default": 1000,
                    "min": 0,
                    "max": 1000,
                    "step": 0.1,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "apply"
    CATEGORY = "loaders"
    
    def apply(self, model, cache_interval, cache_depth, start_step, end_step):
        new_model = model.clone()

        model_step = 0
        current_t = -1
        current_step = -1
        cache_h = None
        encoder_hid_proj = nn.Linear(4096, 2048, bias=True)
        encoder_hid_proj = encoder_hid_proj.to('cuda')

        def apply_model(model_function, kwargs):
            pass

        def apply_model(model_function, kwargs):
            nonlocal current_t, current_step, cache_h, model_step
            
            xa = kwargs["input"]
            t = kwargs["timestep"]
            c_concat = kwargs["c"].get("c_concat", None)
            c_crossattn = kwargs["c"].get("c_crossattn", None)
            y = kwargs["c"].get("y", None)
            control = kwargs["c"].get("control", None)
            transformer_options = kwargs["c"].get("transformer_options", None)

            # https://github.com/comfyanonymous/ComfyUI/blob/629e4c552cc30a75d2756cbff8095640af3af163/comfy/model_base.py#L51-L69
            sigma = t
            xc = new_model.model.model_sampling.calculate_input(sigma, xa)
            if c_concat is not None:
                xc = torch.cat([xc] + [c_concat], dim=1)

            context = c_crossattn
            dtype = new_model.model.get_dtype()
            xc = xc.to(dtype)
            t = new_model.model.model_sampling.timestep(t).float()
            context = context.to(dtype)
            with torch.cuda.amp.autocast(enabled=True):
                context = encoder_hid_proj(context)

            extra_conds = {}
            for o in kwargs:
                extra = kwargs[o]
                if hasattr(extra, "to"):
                    extra = extra.to(dtype)
                extra_conds[o] = extra

            x = xc
            timesteps = t
            y = None if y is None else y.to(dtype)
            transformer_options["original_shape"] = list(x.shape)
            transformer_options["transformer_index"] = 0
            transformer_patches = transformer_options.get("patches", {})

            unet = new_model.model.diffusion_model
            num_video_frames = kwargs.get("num_video_frames",
                                          unet.default_num_video_frames)
            image_only_indicator = kwargs.get("image_only_indicator", None)
            time_context = kwargs.get("time_context", None)

            model_step += 1

            if t[0].item() > current_t:
                model_step = 0
                current_step = -1

            current_t = t[0].item()
            # apply = 1000 - end_step <= current_t <= 1000 - start_step # tは999->0
            apply = start_step <= model_step <= end_step

            if apply:
                current_step += 1
            else:
                current_step = -1
            current_t = t[0].item()

            print(f'model_step: {model_step}, {apply}')

            # https://github.com/comfyanonymous/ComfyUI/blob/629e4c552cc30a75d2756cbff8095640af3af163/comfy/ldm/modules/diffusionmodules/openaimodel.py#L598-L659

            assert (y is not None) == (
                unet.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to(unet.dtype)
            emb = unet.time_embed(t_emb)

            if "emb_patch" in transformer_patches:
                patch = transformer_patches["emb_patch"]
                for p in patch:
                    emb = p(emb, unet.model_channels, transformer_options)

            if unet.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + unet.label_emb(y)

            h = x.type(unet.dtype)

            step_cache_interval = current_step % cache_interval

            for id, module in enumerate(unet.input_blocks):
                transformer_options["block"] = ("input", id)
                h = forward_timestep_embed(
                    module,
                    h,
                    emb,
                    context,
                    transformer_options,
                    time_context=time_context,
                    num_video_frames=num_video_frames,
                    image_only_indicator=image_only_indicator)
                h = apply_control(h, control, 'input')
                if "input_block_patch" in transformer_patches:
                    patch = transformer_patches["input_block_patch"]
                    for p in patch:
                        h = p(h, transformer_options)

                hs.append(h)
                if "input_block_patch_after_skip" in transformer_patches:
                    patch = transformer_patches["input_block_patch_after_skip"]
                    for p in patch:
                        h = p(h, transformer_options)

                # input cache
                if id == cache_depth and apply:
                    if not step_cache_interval == 0:
                        break # cache位置以降はスキップ

            # middle cache
            if step_cache_interval == 0 or not apply:
                transformer_options["block"] = ("middle", 0)
                if unet.middle_block is not None:
                    h = forward_timestep_embed(unet.middle_block, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                h = apply_control(h, control, 'middle')

            for id, module in enumerate(unet.output_blocks):
                if id < len(unet.output_blocks) - cache_depth - 1 and apply:
                    if not step_cache_interval == 0:
                        continue # cache位置以前はスキップ
                
                if id == len(unet.output_blocks) - cache_depth -1 and apply:
                    if step_cache_interval == 0:
                        cache_h = h # cache
                    else:
                        h = cache_h # load cache
                
                transformer_options["block"] = ("output", id)
                hsp = hs.pop()
                hsp = apply_control(hsp, control, 'output')

                if "output_block_patch" in transformer_patches:
                    patch = transformer_patches["output_block_patch"]
                    for p in patch:
                        h, hsp = p(h, hsp, transformer_options)

                h = th.cat([h, hsp], dim=1)
                del hsp
                if len(hs) > 0:
                    output_shape = hs[-1].shape
                else:
                    output_shape = None
                h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)

            h = h.type(x.dtype)
            if unet.predict_codebook_ids:
                model_output =  unet.id_predictor(h)
            else:
                model_output =  unet.out(h)
            
            return new_model.model.model_sampling.calculate_denoised(sigma, model_output, xa)

        new_model.set_model_unet_function_wrapper(apply_model)

        return (new_model, )

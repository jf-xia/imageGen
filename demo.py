import torch
import json
import os
from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline
from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
from diffusers.models.autoencoders import AutoencoderKL
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import FlowMatchEulerDiscreteScheduler
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Boogu-Image-0.1 Turbo Demo")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape of Guilin mountains at golden hour, 8k resolution")
    parser.add_argument("--output", type=str, default="boogu_output.png")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (3-4 recommended for Turbo)")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = "models/Boogu-Image-0.1-Turbo-fp8"

    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using Mac GPU acceleration: {device}")
    elif torch.cuda.is_available():
        device = "cuda:0"
        print(f"Using CUDA: {device}")
    else:
        device = "cpu"
        print(f"Using CPU: {device}")

    if device == "mps":
        dtype = torch.float16
        print(f"Using float16 for MPS compatibility (MPS does not support bfloat16 matmul)")
    elif device != "cpu":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"Loading model components from: {model_path}...")
    start_load = time.time()

    with open(os.path.join(model_path, "model_index.json")) as f:
        config = json.load(f)

    print("  Loading transformer...")
    transformer_dir = os.path.join(model_path, "transformer")
    with open(os.path.join(transformer_dir, "diffusion_pytorch_model.bin.index.json")) as f:
        index = json.load(f)

    state_dict = {}
    shard_files = sorted(set(index["weight_map"].values()))
    for shard_file in shard_files:
        shard_path = os.path.join(transformer_dir, shard_file)
        print(f"    Loading shard: {shard_file}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for k, v in shard.items():
            if hasattr(v, 'tensor_impl'):
                state_dict[k] = v.tensor_impl.dequantize().to(dtype)
            elif hasattr(v, 'dequantize'):
                state_dict[k] = v.dequantize().to(dtype)
            else:
                state_dict[k] = v.to(dtype) if hasattr(v, 'to') else v
        del shard

    with open(os.path.join(transformer_dir, "config.json")) as f:
        transformer_config = json.load(f)

    transformer = BooguImageTransformer2DModel(**transformer_config)
    for name, param in transformer.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name])
        else:
            print(f"    WARNING: {name} not found in checkpoint")
    del state_dict
    print("    Transformer loaded")

    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        os.path.join(model_path, "vae"),
        torch_dtype=dtype,
    )

    print("  Loading MLLM (Qwen3-VL)...")
    mllm = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(model_path, "mllm"),
        torch_dtype=dtype,
    )
    if device == "mps":
        mllm = mllm.to(dtype=torch.float16)

    print("  Loading processor...")
    processor = Qwen3VLProcessor.from_pretrained(
        os.path.join(model_path, "processor"),
    )

    print("  Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        os.path.join(model_path, "scheduler"),
    )

    print("  Constructing pipeline...")
    pipe = BooguImageTurboPipeline(
        transformer=transformer,
        vae=vae,
        mllm=mllm,
        processor=processor,
        scheduler=scheduler,
    )
    pipe.to(device)

    print(f"Model loaded in {time.time() - start_load:.2f}s")

    print(f"Generating image with prompt: {args.prompt}")
    print(f"Steps: {args.steps}, Size: {args.width}x{args.height}")

    start_gen = time.time()

    with torch.no_grad():
        image = pipe(
            instruction=[args.prompt],
            negative_instruction="",
            empty_instruction="",
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            text_guidance_scale=1.0,
            image_guidance_scale=1.0,
            empty_instruction_guidance_scale=0.0,
            use_dmd_student_inference=True,
            dmd_conditioning_sigma=0.001,
            generator=torch.Generator(device).manual_seed(args.seed),
            device=device,
        ).images[0]

    print(f"Image generated in {time.time() - start_gen:.2f}s")

    image.save(args.output)
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()

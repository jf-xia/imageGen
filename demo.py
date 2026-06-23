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


def detect_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def get_dtype(device):
    if device == "mps":
        return torch.float16
    elif device != "cpu":
        return torch.bfloat16
    return torch.float32


def load_model(model_path, device, dtype):
    components = {}
    total_start = time.time()

    def timer(label):
        class Timer:
            def __enter__(self):
                self.start = time.time()
                return self
            def __exit__(self, *args):
                elapsed = time.time() - self.start
                print(f"    [{elapsed:.1f}s]")
        return Timer()

    print("[1/5] Loading transformer (FP8 dequantize)...")
    with timer("transformer"):
        transformer_dir = os.path.join(model_path, "transformer")
        with open(os.path.join(transformer_dir, "diffusion_pytorch_model.bin.index.json")) as f:
            index = json.load(f)

        state_dict = {}
        shard_files = sorted(set(index["weight_map"].values()))
        for i, shard_file in enumerate(shard_files):
            print(f"      Shard {i+1}/{len(shard_files)}: {shard_file}", end=" ", flush=True)
            shard_path = os.path.join(transformer_dir, shard_file)
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            for k, v in shard.items():
                if hasattr(v, 'tensor_impl'):
                    state_dict[k] = v.tensor_impl.dequantize().to(dtype)
                elif hasattr(v, 'dequantize'):
                    state_dict[k] = v.dequantize().to(dtype)
                else:
                    state_dict[k] = v.to(dtype) if hasattr(v, 'to') else v
            del shard
            print(f"({len(state_dict)} weights)")

        with open(os.path.join(transformer_dir, "config.json")) as f:
            transformer_config = json.load(f)

        transformer = BooguImageTransformer2DModel(**transformer_config)
        for name, param in transformer.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name])
        del state_dict
        transformer = transformer.to(dtype)
    components["transformer"] = transformer

    print("[2/5] Loading VAE...")
    with timer("vae"):
        vae = AutoencoderKL.from_pretrained(
            os.path.join(model_path, "vae"), torch_dtype=dtype,
        )
        vae = vae.to(dtype)
    components["vae"] = vae

    print("[3/5] Loading MLLM (Qwen3-VL)...")
    with timer("mllm"):
        mllm = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(model_path, "mllm"), torch_dtype=dtype,
        )
        mllm = mllm.to(dtype)
    components["mllm"] = mllm

    print("[4/5] Loading processor & scheduler...")
    with timer("processor+scheduler"):
        processor = Qwen3VLProcessor.from_pretrained(
            os.path.join(model_path, "processor"),
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            os.path.join(model_path, "scheduler"),
        )
    components["processor"] = processor
    components["scheduler"] = scheduler

    print("[5/5] Constructing pipeline...")
    with timer("pipeline"):
        pipe = BooguImageTurboPipeline(**components)
        pipe.to(device)

    total = time.time() - total_start
    print(f"\nAll components loaded in {total:.1f}s")
    return pipe


def generate(pipe, prompt, device, args):
    print(f"\nGenerating: \"{prompt}\"")
    print(f"  Size: {args.width}x{args.height}, Steps: {args.steps}, Seed: {args.seed}")

    start = time.time()
    with torch.no_grad():
        image = pipe(
            instruction=[prompt],
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

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")
    return image


def main():
    parser = argparse.ArgumentParser(description="Boogu-Image-0.1 Turbo Demo")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--output", type=str, default="boogu_output.png")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=["mps", "cuda:0", "cpu"],
                        help="Device: mps, cuda:0, or cpu (default: auto-detect)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: keep generating from prompts")
    parser.add_argument("--batch", type=str, nargs="+", default=None,
                        help="Batch mode: generate multiple images from prompts")
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), "models", "Boogu-Image-0.1-Turbo-fp8")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run: python download_model.py")
        return

    device = args.device or detect_device()
    dtype = get_dtype(device)
    print(f"Device: {device} | Dtype: {dtype}")

    pipe = load_model(model_path, device, dtype)

    # Batch mode
    if args.batch:
        for i, prompt in enumerate(args.batch):
            image = generate(pipe, prompt, device, args)
            out = args.output.replace(".png", f"_{i}.png") if len(args.batch) > 1 else args.output
            image.save(out)
            print(f"  Saved: {out}")
        return

    # Interactive mode
    if args.interactive:
        print("\nInteractive mode. Type your prompt (Ctrl+C to exit):\n")
        gen = 0
        while True:
            try:
                prompt = input("Prompt> ").strip()
                if not prompt:
                    continue
                image = generate(pipe, prompt, device, args)
                gen += 1
                out = args.output.replace(".png", f"_{gen}.png")
                image.save(out)
                print(f"  Saved: {out}")
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
        return

    # Single generation
    prompt = args.prompt or "A beautiful landscape of Guilin mountains at golden hour, 8k resolution"
    image = generate(pipe, prompt, device, args)
    image.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

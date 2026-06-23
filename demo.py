import torch
import json
import os
from PIL import Image
from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline
from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline
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


def load_transformer(model_path, dtype):
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
    return transformer.to(dtype)


def load_model(model_path, device, dtype, pipeline_cls):
    components = {}
    total_start = time.time()

    def timer(label):
        class Timer:
            def __enter__(self):
                self.start = time.time()
                return self
            def __exit__(self, *args):
                print(f"    [{time.time() - self.start:.1f}s]")
        return Timer()

    print("[1/5] Loading transformer (FP8 dequantize)...")
    with timer("transformer"):
        components["transformer"] = load_transformer(model_path, dtype)

    print("[2/5] Loading VAE...")
    with timer("vae"):
        components["vae"] = AutoencoderKL.from_pretrained(
            os.path.join(model_path, "vae"), torch_dtype=dtype,
        ).to(dtype)

    print("[3/5] Loading MLLM (Qwen3-VL)...")
    with timer("mllm"):
        components["mllm"] = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(model_path, "mllm"), torch_dtype=dtype,
        ).to(dtype)

    print("[4/5] Loading processor & scheduler...")
    with timer("processor+scheduler"):
        components["processor"] = Qwen3VLProcessor.from_pretrained(
            os.path.join(model_path, "processor"),
        )
        components["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(
            os.path.join(model_path, "scheduler"),
        )

    print("[5/5] Constructing pipeline...")
    with timer("pipeline"):
        pipe = pipeline_cls(**components)
        pipe.to(device)

    total = time.time() - total_start
    print(f"\nAll components loaded in {total:.1f}s")
    return pipe


def generate_t2i(pipe, prompt, device, args):
    print(f"\n[Text-to-Image] \"{prompt}\"")
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
            text_guidance_scale=args.guidance,
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


def generate_ti2i(pipe, prompt, input_image_path, device, args):
    print(f"\n[Image Editing] \"{prompt}\"")
    print(f"  Input: {input_image_path}")
    print(f"  Steps: {args.steps}, Seed: {args.seed}")

    input_image = Image.open(input_image_path).convert("RGB")
    print(f"  Image size: {input_image.size[0]}x{input_image.size[1]}")

    start = time.time()
    with torch.no_grad():
        image = pipe(
            instruction=[prompt],
            input_images=[[input_image]],
            negative_instruction="",
            height=args.height or None,
            width=args.width or None,
            max_input_image_pixels=args.max_pixels,
            max_input_image_side_length=args.max_side_length,
            align_res=True,
            num_inference_steps=args.steps,
            text_guidance_scale=args.guidance,
            image_guidance_scale=1.0,
            generator=torch.Generator(device).manual_seed(args.seed),
            device=device,
        ).images[0]

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")
    return image


def main():
    parser = argparse.ArgumentParser(description="Boogu-Image-0.1 Demo")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--output", type=str, default="boogu_output.png")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=["mps", "cuda:0", "cpu"])
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--batch", type=str, nargs="+", default=None)
    parser.add_argument("--edit-image", type=str, default=None,
                        help="Path to input image for editing mode")
    parser.add_argument("--guidance", type=float, default=None,
                        help="Text guidance scale (default: 1.0 for Turbo, 4.0 for Edit)")
    parser.add_argument("--max-pixels", type=int, default=2048*2048,
                        help="Max input image pixels for editing")
    parser.add_argument("--max-side-length", type=int, default=2048*2,
                        help="Max input image side length for editing")
    args = parser.parse_args()

    edit_mode = args.edit_image is not None

    if edit_mode:
        model_name = "Boogu-Image-0.1-Edit-fp8"
        pipeline_cls = BooguImagePipeline
        default_steps = 50
        default_guidance = 4.0
        default_output = "edited_output.png"
    else:
        model_name = "Boogu-Image-0.1-Turbo-fp8"
        pipeline_cls = BooguImageTurboPipeline
        default_steps = 4
        default_guidance = 1.0
        default_output = "boogu_output.png"

    if args.steps is None:
        args.steps = default_steps
    if args.guidance is None:
        args.guidance = default_guidance

    mode_name = "Edit (TI2I)" if edit_mode else "Turbo (T2I)"
    print(f"Mode: {mode_name}")
    print(f"Model: {model_name}")

    model_path = os.path.join(os.path.dirname(__file__), "models", model_name)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print(f"Run: python -c \"from huggingface_hub import snapshot_download; snapshot_download('Boogu/{model_name}', local_dir='{model_path}')\"")
        return

    device = args.device or detect_device()
    dtype = get_dtype(device)
    print(f"Device: {device} | Dtype: {dtype}")

    pipe = load_model(model_path, device, dtype, pipeline_cls)

    if args.batch:
        for i, prompt in enumerate(args.batch):
            if edit_mode:
                image = generate_ti2i(pipe, prompt, args.edit_image, device, args)
            else:
                image = generate_t2i(pipe, prompt, device, args)
            out = args.output.replace(".png", f"_{i}.png") if len(args.batch) > 1 else args.output
            image.save(out)
            print(f"  Saved: {out}")
        return

    if args.interactive:
        mode_hint = "[editing] provide image path + prompt" if edit_mode else "[text-to-image]"
        print(f"\nInteractive mode {mode_hint}. Type your prompt (Ctrl+C to exit):\n")
        gen = 0
        while True:
            try:
                if edit_mode:
                    img_path = input("Image> ").strip()
                    if not img_path:
                        continue
                    if not os.path.exists(img_path):
                        print(f"  Error: {img_path} not found")
                        continue
                    prompt = input("Prompt> ").strip()
                else:
                    prompt = input("Prompt> ").strip()
                if not prompt:
                    continue

                if edit_mode:
                    image = generate_ti2i(pipe, prompt, img_path, device, args)
                else:
                    image = generate_t2i(pipe, prompt, device, args)

                gen += 1
                out = args.output.replace(".png", f"_{gen}.png")
                image.save(out)
                print(f"  Saved: {out}")
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
        return

    if edit_mode:
        if not args.edit_image or not os.path.exists(args.edit_image):
            print(f"Error: Input image not found: {args.edit_image}")
            return
        prompt = args.prompt or "Edit this image"
        image = generate_ti2i(pipe, prompt, args.edit_image, device, args)
    else:
        prompt = args.prompt or "A beautiful landscape of Guilin mountains at golden hour, 8k resolution"
        image = generate_t2i(pipe, prompt, device, args)

    image.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

import os
import torch
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import numpy as np


def load_state_dict(model, checkpoint, strict=True):
    """Load state dict from checkpoint"""
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=strict)


def load_image(image_path):
    """Load image without resizing - use original dimensions"""
    image = Image.open(image_path).convert('RGB')
    
    # Convert to tensor and normalize without resizing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension


def prepare_data_dict(lr_image, ref_image, hr_image, device):
    """Prepare data dictionary for model input"""
    data_dict = {
        "lr": lr_image.to(device),
        "ref": ref_image.to(device),
    }
    
    if hr_image is not None:
        data_dict["hr"] = hr_image.to(device)
    
    return data_dict


def save_tensor_as_image(tensor, output_path):
    """Save tensor as image"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    tensor = tensor.cpu()
    image_array = tensor.permute(1, 2, 0).numpy()
    image_array = (image_array * 255).astype(np.uint8)
    
    image = Image.fromarray(image_array)
    image.save(output_path)
    print(f"Saved image to {output_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # Model and checkpoint
    parser.add_argument("--ckpt", required=True, type=str, help="checkpoint path")
    parser.add_argument("--config", default="configs/model/refsr_dino.yaml", type=str, help="model config path")
    
    # Input images
    parser.add_argument("--lr_image", required=True, type=str, help="Low resolution image path")
    parser.add_argument("--hr_image", type=str, help="High resolution image path (optional, for comparison)")
    parser.add_argument("--ref_image", required=True, type=str, help="Reference image path")
    
    # Output
    parser.add_argument("--output", type=str, default="./single_inference_result", help="output directory")
    
    # Settings
    parser.add_argument("--global_ref_scale", type=float, default=1.0, help="global scalar scaling factor for reference")
    parser.add_argument("--local_ref", action="store_true", help="Whether to use local reference scaling")
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cpu", "cuda", "mps"])
    
    return parser.parse_args()


def single_image_inference(args):
    """Perform inference on a single set of images"""
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model configuration
    print(f"Loading model configuration from {args.config}")
    model_config = OmegaConf.load(args.config)
    model = instantiate_from_config(model_config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    load_state_dict(model, checkpoint, strict=True)
    
    # Setup model
    model.freeze()
    model.to(args.device)
    model.eval()
    
    # Load images at their original sizes
    print("Loading images...")
    lr_image = load_image(args.lr_image)      # Original size (should be 48x48)
    ref_image = load_image(args.ref_image)    # Original size (should be 800x800)
    
    hr_image = None
    if args.hr_image:
        hr_image = load_image(args.hr_image)  # Original size (should be 800x800)
    
    print(f"LR image shape: {lr_image.shape}")
    print(f"Ref image shape: {ref_image.shape}")
    if hr_image is not None:
        print(f"HR image shape: {hr_image.shape}")
    
    # Prepare data dictionary
    data_dict = prepare_data_dict(lr_image, ref_image, hr_image, args.device)
    
    # Setup reference scaling
    if args.global_ref_scale != 1.0:
        sim_lamuda = args.global_ref_scale
        print(f"Using global reference scaling: {sim_lamuda}")
    elif args.local_ref:
        # Use the actual reference image dimensions for local scaling
        _, _, h, w = ref_image.shape
        sim_lamuda = torch.ones((h, w)).to(args.device)
        print(f"Using local reference scaling with shape: {sim_lamuda.shape}")
    else:
        sim_lamuda = None
        print("No reference scaling applied")
    
    # Perform inference
    print("Starting inference...")
    with torch.no_grad():
        if sim_lamuda is not None:
            # If using reference scaling
            if hasattr(model, 'sample_with_ref_scale'):
                output = model.sample_with_ref_scale(data_dict, sim_lamuda)
            else:
                # Fallback method
                output = model.sample(data_dict)
        else:
            # Standard inference
            output = model.sample(data_dict)
    
    # Save results
    print("Saving results...")
    
    # Save super-resolved image
    if isinstance(output, dict) and 'samples' in output:
        sr_image = output['samples']
    else:
        sr_image = output
    
    sr_output_path = os.path.join(args.output, "super_resolved.png")
    save_tensor_as_image(sr_image, sr_output_path)
    
    # Save input images for comparison
    lr_output_path = os.path.join(args.output, "input_lr.png")
    save_tensor_as_image(lr_image, lr_output_path)
    
    ref_output_path = os.path.join(args.output, "input_ref.png")
    save_tensor_as_image(ref_image, ref_output_path)
    
    if hr_image is not None:
        hr_output_path = os.path.join(args.output, "input_hr.png")
        save_tensor_as_image(hr_image, hr_output_path)
    
    print(f"Inference completed! Results saved to {args.output}")
    
    # Calculate and print PSNR if HR image is available
    if hr_image is not None:
        try:
            # Calculate PSNR
            mse = torch.mean((sr_image - hr_image) ** 2)
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # Assuming normalized to [-1, 1]
            print(f"PSNR: {psnr.item():.2f} dB")
        except Exception as e:
            print(f"Could not calculate PSNR: {e}")


def main():
    args = parse_args()
    
    # Validate input files
    if not os.path.exists(args.lr_image):
        raise FileNotFoundError(f"LR image not found: {args.lr_image}")
    if not os.path.exists(args.ref_image):
        raise FileNotFoundError(f"Reference image not found: {args.ref_image}")
    if args.hr_image and not os.path.exists(args.hr_image):
        raise FileNotFoundError(f"HR image not found: {args.hr_image}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    print("=" * 50)
    print("Reference-based Super Resolution - Single Image Inference")
    print("=" * 50)
    print(f"LR Image: {args.lr_image}")
    print(f"Reference Image: {args.ref_image}")
    print(f"HR Image: {args.hr_image if args.hr_image else 'Not provided'}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Config: {args.config}")
    print(f"Output Directory: {args.output}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # Run inference
    single_image_inference(args)


if __name__ == "__main__":
    main()

# how to run:
# python single_inference.py \
#     --ckpt "step=372999-val_psnr=20.864.ckpt" \
#     --lr_image "path/to/your/48x48_image.jpg" \
#     --ref_image "path/to/your/800x800_reference.jpg" \
#     --local_ref \
#     --output "./results"

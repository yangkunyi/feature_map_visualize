import argparse
import os
import sys
from vitWrapper import ViTWrapper
from denoiser import Denoiser
from PIL import Image
import torchvision.transforms as T
from utlis import get_denoised_features, get_features
import re
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("extract and visualize the features")
    parser.add_argument(
        "--vit_type",
        default="vit_base_patch14_dinov2.lvd142m",
        type=str,
    )
    parser.add_argument(
        "--denoised",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--vit_stride", default=14, type=int, help="patch resolution of the self.model."
    )
    parser.add_argument(
        "--n_clusters", default=10, type=int, help="number of kmeans clusters."
    )
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument(
        "--resize",
        default=None,
        type=int,
        nargs="+",
        help="""Apply a resize transformation to input image(s).
        Usage (single or H W): --resize 512, --resize 720 1280""",
    )
    return parser.parse_args()

def get_patch_size(model_type:str):

    pattern = r"patch(\d+)"

    match = re.search(pattern, model_type)

    return int(match.group(1))



def run(args):
    if os.path.exists(args.input_path):
        patch_size = get_patch_size(args.vit_type)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        
        if os.path.isfile(args.input_path):

            if args.output_dir == None:
                output_folder = os.path.dirname(args.input_path)
            else:
                output_folder = args.output_dir

            sample_img_path = args.input_path

        else:
            if args.output_dir == None:
                output_folder = args.input_path
            else:
                output_folder = args.output_dir
                
            entries = os.listdir(args.input_path)
            file_list = []
            for entrie in entries:
                file_list.append(os.path.join(args.input_path, entrie))

            sample_img_path = file_list[0]


        os.makedirs(output_folder, exist_ok=True)
        img = Image.open(sample_img_path).convert('RGB')


        if args.denoised == True:
            assert args.vit_type in [
                # DINOv1
                "vit_base_patch16_224.dino",
                # DINOv2
                "vit_base_patch14_dinov2.lvd142m",
            ]
            
            vit = ViTWrapper(args.vit_type,stride=patch_size)

            if patch_size == 14:
                img_size = (518,518)
                model = Denoiser(noise_map_height=37, noise_map_width=37,vit=vit,feature_dim=768)
                freevit_model_ckpt = torch.load('/bd_byta6000i0/users/dataset/feat_visualize_models/dinov2_v1.pth')["denoiser"]
                model.load_state_dict(freevit_model_ckpt,strict=False)
            else:
                img_size = (512,512)
                model = Denoiser(noise_map_height=32, noise_map_width=32,vit=vit,feature_dim=768)
                freevit_model_ckpt = torch.load('/bd_byta6000i0/users/dataset/feat_visualize_models/dino_v1.pth')["denoiser"]
                model.load_state_dict(freevit_model_ckpt,strict=False)
            
        else:
            if args.resize == None:
                W,H = img.size
                # img_size = (H // patch_size * patch_size, W // patch_size * patch_size)
            else:
                try:
                    H = args.resize[0]
                    W = args.resize[1]
                except:
                    H = args.resize[0]
                    W = args.resize[0]
            img_size = (H // patch_size * patch_size, W // patch_size * patch_size)
            model = ViTWrapper(args.vit_type,stride=patch_size)

        
        model.eval()
        model.to(device)

        transform = T.Compose([
            T.Resize(img_size,T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
        ])

        # deal with one file
        if os.path.isfile(args.input_path):
            img = transform(img)[:3].unsqueeze(0)
            img = img.to(device)

            with torch.no_grad():
                if args.denoised == True:
                    features = get_denoised_features(img, model, img_size, args.n_clusters)
                else: 
                    features = get_features(img, model, img_size, args.n_clusters)
            to_pic = T.ToPILImage()


            full_file_name = os.path.basename(args.input_path)
            file_name, file_ext = os.path.splitext(full_file_name)
            outfile = f'{file_name}_features{file_ext}'

            visual_features = to_pic(features)
            visual_features.save(os.path.join(output_folder, outfile))
        
        # deal with the whole dir
        else:
            for img_path in file_list:
                img = Image.open(img_path).convert('RGB')
                img = transform(img)[:3].unsqueeze(0)
                img = img.to(device)

                with torch.no_grad():
                    if args.denoised == True:
                        features = get_denoised_features(img, model, img_size, args.n_clusters)
                    else: 
                        features = get_features(img, model, img_size, args.n_clusters)

                to_pic = T.ToPILImage()

                full_file_name = os.path.basename(img_path)
                file_name, file_ext = os.path.splitext(full_file_name)
                outfile = f'{file_name}_features{file_ext}'

                visual_features = to_pic(features)
                visual_features.save(os.path.join(output_folder, outfile))

    
    else:
        print(f"Provided input path {args.input_path} doesn't exists.")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    run(args)
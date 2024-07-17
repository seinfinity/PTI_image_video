import torch
import dnnlib
import os
import matplotlib.pyplot as plt
from PIL import Image
from configs import paths_config
from latent_creators import sg2_latent_creator
from models.e4e.stylegan2.model import Generator
import legacy

# GPU 사용 가능 여부 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def create_latents(image_path):
    # SG2 Latent 생성
    sg2_latent_creator_instance = sg2_latent_creator.SG2LatentCreator(projection_steps=600)
    sg2_latent_creator_instance.create_latents()

def load_latents(image_name, w_path_dir, latent_type):
    print("############################## start to load latents ##############################")
    embedding_dir = f'{w_path_dir}/{latent_type}/{image_name}'
    inversions = torch.load(f'{embedding_dir}/0.pt')
    return inversions

def mix_latents(target_latent, source_latent, target_ratio=0.3, source_ratio=0.7):
    return target_ratio * target_latent + source_ratio * source_latent

def get_image_from_w(w, G):
    if len(w.size()) <= 2:
        w = w.unsqueeze(0)
    img = G.synthesis(w, noise_mode='const', force_fp32=True)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    return img[0]

def plot_image_from_w(w, G):
    img = get_image_from_w(w, G)
    plt.axis('off')
    resized_image = Image.fromarray(img, mode='RGB').resize((256, 256))
    plt.imshow(resized_image)
    plt.show()

def main(frame_dir, source_img_path, output_dir, w_path_dir, generator_path, latent_type):
    create_latents(source_img_path)
    
    # 1. Load latents for source image
    source_image_name = os.path.basename(source_img_path).split('.')[0]
    source_latents = load_latents(source_image_name, w_path_dir, latent_type)

    # 2. Load generator
    print(f"Loading generator from path: {generator_path}")
    if not os.path.isfile(generator_path):
        print(f"File not found: {generator_path}")
        return
    
    generator = Generator(1024, 512, 8).to(device)  # 해상도 및 스타일 코드 크기에 맞게 수정
    try:
        with dnnlib.util.open_url(generator_path) as f:
            generator = legacy.load_network_pkl(f)['G_ema'].to(device)
    except Exception as e:
        print(f"Error loading the generator: {e}")
        return

    # 3. Process each frame
    for frame_file in sorted(os.listdir(frame_dir)):
        if frame_file.endswith(".png"):
            # Load latents for current frame
            frame_name = os.path.splitext(frame_file)[0]
            target_latents = load_latents(frame_name, w_path_dir, latent_type)

            # Mix latents with defined ratios
            mixed_latent = mix_latents(target_latents, source_latents, target_ratio=0.3, source_ratio=0.7)

            # Generate and save mixed image
            mixed_image = get_image_from_w(mixed_latent, generator)
            mixed_image_pil = Image.fromarray(mixed_image)
            output_img_path = os.path.join(output_dir, f"output_{frame_name}.jpg")
            mixed_image_pil.save(output_img_path)

            # Plot mixed image (optional)
            plot_image_from_w(mixed_latent, generator)

if __name__ == "__main__":
    frame_dir = os.path.join(paths_config.input_data_path, "frames")
    source_img_path = paths_config.input_data_path + "/source.jpg"
    output_dir = paths_config.output_data_path  # 수정 필요
    w_path_dir = paths_config.embedding_base_dir  # 수정 필요
    generator_path = paths_config.stylegan2_ada_ffhq  # generator 경로 설정 (stylegan2 설정)
    latent_type = paths_config.sg2_results_keyword  # 사용할 latent type 선택 (sg2, e4e, sg2_plus 중 하나)
    main(frame_dir, source_img_path, output_dir, w_path_dir, generator_path, latent_type)

import torch
import dnnlib
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
from configs import paths_config
from latent_creators import e4e_latent_creator, sg2_latent_creator, sg2_plus_latent_creator
from models.stylegan2.model import Generator  # 필요한 경우 수정
# import legacy

# GPU 사용 가능 여부 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_latents(image_name, w_path_dir):
    inversions = {}
    sg2_embedding_dir = f'{w_path_dir}/{paths_config.sg2_results_keyword}/{image_name}'
    inversions[paths_config.sg2_results_keyword] = torch.load(f'{sg2_embedding_dir}/0.pt')
    e4e_embedding_dir = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}'
    inversions[paths_config.e4e_results_keyword] = torch.load(f'{e4e_embedding_dir}/0.pt')
    sg2_plus_embedding_dir = f'{w_path_dir}/{paths_config.sg2_plus_results_keyword}/{image_name}'
    inversions[paths_config.sg2_plus_results_keyword] = torch.load(f'{sg2_plus_embedding_dir}/0.pt')
    return inversions

def mix_latents(target_latent, source_latent, target_ratio=0.7, source_ratio=0.3):
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

def main(target_img_path, source_img_path, output_img_path, w_path_dir, generator_path):
    # 2. Load latents
    target_image_name = target_img_path.split('/')[-1].split('.')[0]
    source_image_name = source_img_path.split('/')[-1].split('.')[0]
    target_latents = load_latents(target_image_name, w_path_dir)
    source_latents = load_latents(source_image_name, w_path_dir)

    # 3. Mix latents
    mixed_latent = mix_latents(target_latents[paths_config.sg2_results_keyword], source_latents[paths_config.sg2_results_keyword])

    # 4. Load generator
    print(f"Loading generator from path: {generator_path}")
    if not os.path.isfile(generator_path):
        print(f"File nor found: {generator_path}")
        return
    
    generator = Generator(1024, 512, 8).to(device)  # 해상도 및 스타일 코드 크기에 맞게 수정
    try:
        with dnnlib.util.open_url(generator_path) as f:
            generator = legacy.load_network_pkl(f)['G_ema'].to(device)
    except Exception as e:
        print(f"Error loading the generator: {e}")
        return

    # 5. Generate and save mixed image
    mixed_image = get_image_from_w(mixed_latent, generator)
    mixed_image_pil = Image.fromarray(mixed_image)
    mixed_image_pil.save(output_img_path)

    # 6. Plot mixed image
    plot_image_from_w(mixed_latent, generator)

if __name__ == "__main__":
    target_img_path = paths_config.input_data_path + "/target.jpg"
    source_img_path = paths_config.input_data_path + "/source.jpg"
    output_img_path = paths_config.output_data_path + "/create_output_img.jpg"
    w_path_dir = paths_config.embedding_base_dir  # 수정 필요
    generator_path = paths_config.stylegan2_ada_ffhq  # generator 경로 설정 (stylegan2 설정)
    main(target_img_path, source_img_path, output_img_path, w_path_dir, generator_path)

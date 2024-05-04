import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

base_dir = "./strain_png"
sub_dirs = ["S", "elastography_region_from_grayscale", "elastography_region"]
score_dirs = ["score1", "score2", "score3", "score4", "score5"]

for score in score_dirs:
    output_dir = f"./strain_png/plot_images/{score}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

for score_dir in score_dirs:
    reference_path = os.path.join(base_dir, sub_dirs[0], score_dir)
    image_files = [f for f in os.listdir(reference_path) if f.endswith('.png')]

    for img_file in image_files:
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 2], hspace=0, wspace=0.03)

        img_path = os.path.join(base_dir, sub_dirs[0], score_dir, img_file)
        img = mpimg.imread(img_path)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.imshow(img)
        ax1.axis('off')

        for i in range(1, 3):
            img_path = os.path.join(base_dir, sub_dirs[i], score_dir, img_file)
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                ax = fig.add_subplot(gs[1, i-1])
                ax.imshow(img)
                ax.axis('off')

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.03, hspace=0)

        save_path = os.path.join(f"./strain_png/plot_images/{score_dir}", img_file)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

print("모든 이미지 처리가 완료되었습니다.")
import os
import shutil
import re

def organize_files(base_dir):
    categories = ['G', 'S']
    scores = ['score1', 'score2', 'score3', 'score4', 'score5']
    
    for category in categories:
        for score in scores:
            os.makedirs(os.path.join(base_dir, category, score), exist_ok=True)
    
    for score_dir in scores:
        current_dir = os.path.join(base_dir, score_dir)
        
        if os.path.isdir(current_dir):
            for filename in os.listdir(current_dir):
                if filename.lower().endswith('.png'):
                    match = re.search("[a-zA-Z]", filename)
                    if match:
                        first_alpha = match.group(0).upper()
                        source_path = os.path.join(current_dir, filename)
                        
                        if first_alpha == 'G' or first_alpha == 'S':
                            dest_path = os.path.join(base_dir, first_alpha, score_dir, filename)
                            shutil.copy(source_path, dest_path)
                            print(f"Copied {source_path} to {dest_path}")


base_directory = './strain_png'
organize_files(base_directory)
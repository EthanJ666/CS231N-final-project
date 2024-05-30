import os
import torch
import torch.nn.functional as F

def preprocess_and_save(root_dir, save_dir, new_size=(224, 224)):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    processed_count = 0    

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pt'):
                img_path = os.path.join(subdir, file)
                image = torch.load(img_path)  # Load the .pt file
                
                if len(image.shape) == 4 and image.shape[1] == 3:  # Ensure it has the expected dimensions
                    # Resize the height and width dimensions only
                    # image shape: [frames, channels, height, width]
                    resized_frames = F.interpolate(image, size=new_size, mode='bilinear', align_corners=False)
                    # print("resized_frames:", resized_frames.shape)  # Debugging line
                else:
                    print(f'Skipping {img_path} as it does not match expected dimensions')
                    continue

                # Create the corresponding subdir in the save_dir
                relative_path = os.path.relpath(subdir, root_dir)
                save_subdir = os.path.join(save_dir, relative_path)
                if not os.path.exists(save_subdir):
                    os.makedirs(save_subdir)

                save_path = os.path.join(save_subdir, file)  # Save the .pt file with the same name
                torch.save(resized_frames, save_path)
                # print(f'Saved {save_path}')

                processed_count += 1
                if processed_count % 100 == 0:
                    print(f'Processed {processed_count} files')

# Usage example
# root_dir = './dataset_images'
# save_dir = './dataset_images_resize'
# preprocess_and_save(root_dir, save_dir)

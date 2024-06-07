import os
import cv2

def crop_image(image, rows, cols):
    h, w = image.shape[:2]
    crop_h = h // rows
    crop_w = w // cols
    crops = []
    for i in range(rows):
        for j in range(cols):
            start_y = i * crop_h
            start_x = j * crop_w
            end_y = start_y + crop_h
            end_x = start_x + crop_w
            crop = image[start_y:end_y, start_x:end_x]
            crops.append((i, j, crop))
    return crops

def process_directory(input_dir, output_dir, rows, cols):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            crops = crop_image(image, rows, cols)
            for idx, (i, j, crop) in enumerate(crops):
                crop_filename = f'{os.path.splitext(filename)[0]}_crop_{i}_{j}.jpg'
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

base_input_directory = '/local_datasets/ASD/All_ver2/03/train'
base_output_directory = '/local_datasets/ASD/All_ver2/03/train_cropped'

classes = ['ASD', 'TD']

for cls in classes:
    input_directory = os.path.join(base_input_directory, cls)
    output_directory = os.path.join(base_output_directory, cls)
    print("Starting process")
    process_directory(input_directory, output_directory, 3, 3)  # N x M crop

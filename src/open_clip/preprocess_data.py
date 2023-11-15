import json
import cv2
import numpy as np
import os
from tqdm import tqdm

def extract_data(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return [(d['question'], d['answer'], d['video_link']) for d in data]

def extract_frames(num_frames, video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in tqdm(range(num_frames), desc='Extracting frames', leave=False, ascii=True):
        ret, frame = cap.read()
        frames.append(frame)
    cap.release()
    return frames

def concatenate_images(frames):
    return np.concatenate([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames], axis=1)

# def extract_data_and_concat(json_directory, images_path, qa_path):
#     os.makedirs(images_path, exist_ok=True)
#     os.makedirs(qa_path, exist_ok=True)
# 
#     json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]
#     for json_file in tqdm(json_files, desc='Processing JSON files', ascii=True):
#         json_path = os.path.join(json_directory, json_file)
#         data_list = extract_data(json_path)
# 
#         for i, data in enumerate(tqdm(data_list, desc='Processing data points', leave=False, ascii=True)):
#             frames = extract_frames(9, data[2])
#             concat = concatenate_images(frames)
# 
#             image_filename = f"{os.path.splitext(json_file)[0]}_{i}.png"
#             image_filepath = os.path.join(images_path, image_filename)
#             cv2.imwrite(image_filepath, cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))
# 
#             qa_filename = f"{os.path.splitext(json_file)[0]}_qa.txt"
#             qa_filepath = os.path.join(qa_path, qa_filename)
#             with open(qa_filepath, 'a') as qa_file:
#                 qa_file.write(f"{data[0]} {data[1]}\n")

def extract_data_and_concat(json_directory, images_path, qa_path):
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(qa_path, exist_ok=True)

    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]
    for json_file in tqdm(json_files, desc='Processing JSON files', ascii=True):
        json_path = os.path.join(json_directory, json_file)
        base_filename = os.path.splitext(json_file)[0]
        qa_filename = f"{base_filename}_qa.txt"
        qa_filepath = os.path.join(qa_path, qa_filename)

        if os.path.exists(qa_filepath):
            continue  # Skip already processed JSON file

        data_list = extract_data(json_path)

        for i, data in enumerate(tqdm(data_list, desc='Processing data points', leave=False, ascii=True)):
            image_filename = f"{base_filename}_{i}.png"
            image_filepath = os.path.join(images_path, image_filename)

            if os.path.exists(image_filepath):
                continue  # Skip already created image

            frames = extract_frames(9, data[2])
            if not frames:
                continue  # Skip if video frames are not available

            concat = concatenate_images(frames)
            cv2.imwrite(image_filepath, cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))

            with open(qa_filepath, 'a') as qa_file:
                qa_file.write(f"{data[0]} {data[1]}\n")

def main():
    extract_data_and_concat(
        '/nfs/turbo/coe-mihalcea/namhokoh/openclip_dev/src/wildqadata',
        '/nfs/turbo/coe-mihalcea/namhokoh/openclip_dev/src/all_in_one_images',
        '/nfs/turbo/coe-mihalcea/namhokoh/openclip_dev/src/all_in_one_txt'
    )

if __name__ == "__main__":
    main()
import cv2
import os

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # FPS ni olish

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    return fps  # FPS ni qaytarish

# Videodan ramkalarni ajratish
video_path = 'H.mp4'
frames_output_folder = 'extracted_frame'
fps = extract_frames(video_path, frames_output_folder)

# Videodan audioni ajratish
os.system(f'ffmpeg -i {video_path} -q:a 0 -map a audio.mp3')

# Ramkalarni qayta ishlash
os.system(f'python test_voodoo3d.py --source_root {frames_output_folder} --config_path configs/lp3d.yml --model_path pretrained_models/voodoo3d.pth --save_root results/lp3d_test --cam_batch_size 8 --fps {fps} --skip_preprocess')

# Qayta ishlangan videoga audioni qo'shish
os.system('ffmpeg -i results/lp3d_test/all_images_video.mp4 -i audio.mp3 -c:v copy -c:a aac -strict experimental final_video_with_audio.mp4')





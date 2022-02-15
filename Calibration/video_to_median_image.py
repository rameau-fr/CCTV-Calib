import cv2
import numpy as np

def median_image_video(vid_path, rdn_nb_im):
    cap = cv2.VideoCapture(vid_path)

    # Randomly select n frames
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=rdn_nb_im)

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)

    # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)  

    return medianFrame

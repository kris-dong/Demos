import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from ultralytics import YOLO

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2



def load_model(encoder, device):
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]}
    }
    # depth_anything = DepthAnythingV2(**model_configs[encoder])
    # depth_anything.load_state_dict(torch.load(f"../checkpoints/depth_anything_v2_{encoder}.pth", map_location="cpu"))
    depth_anything = DepthAnythingV2(**{**model_configs[encoder], "max_depth": 10})
    depth_anything.load_state_dict(torch.load(f"./checkpoints/depth_anything_v2_metric_hypersim_{encoder}.pth", map_location="cpu"))
    depth_anything = depth_anything.to(device).eval()

    return depth_anything


def get_depth_image(raw_frame, center_reference, depth_anything, input_size):
    # depth = depth_anything.infer_image(raw_frame, input_size)
    image, (h, w) = depth_anything.image2tensor(raw_frame, input_size)
    depth: torch.Tensor = depth_anything.forward(image)
    depth = torch.nn.functional.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

    return depth

def get_depth_image_visual(depth: torch.Tensor, grayscale: bool):
    # create a copy of the depth tensor on cpu
    depth_img = depth.detach().cpu().numpy()
    
    if grayscale:
        depth_img = depth_img * 100.0
        depth_img = np.clip(depth_img, 0, 255)
        depth_img = depth_img.astype(np.uint8)
        depth_img = np.repeat(depth_img[..., np.newaxis], 3, axis=-1)

    else:
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        depth_img = (cmap(depth_img)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    return depth_img



def path_planner(depth_image: torch.Tensor, side_ratio: float = 0.25, safe_distance: float = 0.5, beta: float = 0.6):
    height, width = depth_image.shape[:2]

    side_width = int(width * side_ratio)
    
    average_distances = torch.zeros(3)
    min_distances = torch.zeros(3)
    threshold = torch.zeros(3, dtype=torch.bool)

    # left side
    left_side = depth_image[:, :side_width]
    average_distances[0] = torch.mean(left_side)
    min_distances[0] = torch.min(left_side)
    # right side
    right_side = depth_image[:, width - side_width:]
    average_distances[2] = torch.mean(right_side)
    min_distances[2] = torch.min(right_side)
    
    # center
    center = depth_image[:, side_width:width - side_width]
    average_distances[1] = torch.mean(center)
    min_distances[1] = torch.min(center)

    threshold = (beta * min_distances + (1-beta) * average_distances) < safe_distance

    return average_distances, threshold


if __name__ == "__main__":
    input_size = 518
    encoder = "vits"
    pred_only = True
    grayscale = True
    
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    depth_anything = load_model(encoder, DEVICE)

    # Load a model
    yolo_model = YOLO("./checkpoints/yolo11l.pt")  # pretrained YOLO11n model

    
    
    # Change to read from camera feed
    video_source = 0  # Use 0 for the default camera, or change to another index for other cameras
    raw_video = cv2.VideoCapture(video_source)  # Initialize camera feed
    
    # Check if the camera opened successfully
    if not raw_video.isOpened():
        print("Error: Could not open video source.")
        exit()
    



    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")


    # show result in a window
    cv2.namedWindow("Depth Anything V2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth Anything V2", 1280, 720)


    goal = torch.zeros(3)


    while True:  # Loop for a maximum number of frames
        ret, raw_frame = raw_video.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        

        center_reference = .45

        # Display the raw frame for debugging
        depth = get_depth_image(raw_frame, center_reference, depth_anything, input_size)
        depth_img = get_depth_image_visual(depth, grayscale)
    


        # do path planning

        side_ratio = 0.25

        average_distances, threshold = path_planner(depth, side_ratio=side_ratio)
        
        if threshold[0]:
            # move to the right
            goal[1] = -0.2
        elif threshold[2]:
            # move to the left
            goal[1] = 0.2
        else:
            goal[1] = 0

        if threshold[1]:
            # move forward
            goal[0] = 0
        else:
            goal[0] = 0.2
            

        # draw lines separating the slices
        side_width = int(depth_img.shape[1] * side_ratio)
        cv2.line(depth_img, (side_width, 0), (side_width, depth_img.shape[0]), (255, 255, 255), 2)
        cv2.line(depth_img, (depth_img.shape[1] - side_width, 0), (depth_img.shape[1] - side_width, depth_img.shape[0]), (255, 255, 255), 2)
        
        # highlight the slice with the lowest average distance
        if threshold[0]:
            cv2.rectangle(depth_img, (0, 0), (side_width, depth_img.shape[0]), (0, 0, 255), 2)
        if threshold[1]:
            cv2.rectangle(depth_img, (side_width, 0), (depth_img.shape[1] - side_width, depth_img.shape[0]), (0, 0, 255), 2)
        if threshold[2]:
            cv2.rectangle(depth_img, (depth_img.shape[1] - side_width, 0), (depth_img.shape[1], depth_img.shape[0]), (0, 0, 255), 2)


        if pred_only:
            cv2.imshow("Depth Anything V2", depth_img)
        else:
            split_region = np.ones((raw_frame.shape[0], margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth_img])
            
            cv2.imshow("Depth Anything V2", combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


            
        # Run batched inference on a list of images
        results = yolo_model.predict(raw_frame, conf=0.2, classes=[46], verbose=False)

        result = results[0]

        # Process results list
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen

        if boxes:
            # get the highest probability class
            highest_idx = torch.argmax(boxes.conf)

            conf = boxes.conf[highest_idx]
            xywhn = boxes[highest_idx].xywhn

            goal[2] = xywhn[0, 0] - 0.5

        annotated_frame = result.plot()



        cv2.imshow("YOLO", annotated_frame)
        cv2.waitKey(1)

        print(goal)

    
    raw_video.release()
    cv2.destroyAllWindows()
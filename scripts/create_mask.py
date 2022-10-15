
import cv2 
import numpy as np
from models.keras_ocr import tools

def mask(path, detector):

    images = [path]
    images = [tools.read(image) for image in images]
    images = [tools.resize_image(image,max_scale=2,max_size=2048) for image in images]
    max_height, max_width = np.array(
                [image.shape[:2] for image, scale in images]
            ).max(axis=0)
    scales = [scale for _, scale in images]
    images = np.array(
            [
                tools.pad(image, width=max_width, height=max_height)
                for image, _ in images
            ]
        )
    h, w, _ = images[0].shape
    # print(images.shape)
    boxes=detector.detect(images)
    masks = np.zeros((1, h, w, 3), dtype='uint8') 
    
    # print(mask.shape)
    index = 0
    org = []
    sizes = []
    for box in boxes:
        for x in range(len(box)):
        
            minXL= min(box[x][0][0],box[x][3][0])
            maxXR= max(box[x][1][0],box[x][2][0])
            minYL= min(box[x][0][1], box[x][1][1])
            maxYU= max(box[x][2][1], box[x][3][1])

            org.append([minXL, maxYU])
            sizes.append(abs(maxYU - minYL)) 
            image_crop = (images[0])[int(minYL):int(maxYU), int(minXL):int(maxXR)]
    
            cv2.imwrite(f'./crop/image_crop_{index}.jpg', image_crop)
            index += 1

            for i in range(int(minXL), int(maxXR)):
                for j in range(int(minYL), int(maxYU)):
                    images[0][j,i]= (255,255,255)
                    masks[0][j,i]= (255,255,255)

            mask= cv2.resize(masks[0], (w//2, h//2), interpolation = cv2.INTER_AREA)     
            image= cv2.resize(images[0], (w//2, h//2), interpolation = cv2.INTER_AREA)
        
    cv2.imwrite(f'./Masks/mask.jpg', mask)
    cv2.imwrite(f'./Images/image.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print('Mask and Image are generated')

    return [org[0][0]//2, org[0][1]//2], max(sizes)//2
   

        
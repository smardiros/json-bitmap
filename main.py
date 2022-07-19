import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageColor
from joblib import Parallel, delayed
import time

img_folder = "panoptic_val2017/"
json_folder = "image_json/" # folder for per-image json outputs
short_file = "short_file.json"
full_file = "panoptic_val2017.json"

#img_info is from the "images" list in the original json
def create_img_json(img_info):
    img_name = img_info["file_name"]
    imp_name_png = img_name.split(".")[0] + ".png"
    img_location = img_folder + imp_name_png
    img_name_json = img_name.split(".")[0] + ".json"

    img_dict = {
        "id": img_info["id"],
        "file_name": img_info["file_name"],
        "flickr_url": img_info["flickr_url"],
        "width": img_info["width"],
        "height": img_info["height"],
        "segments_info": []
    }

    img_json = json.dumps(img_dict, indent=4)

    with open(json_folder + img_name_json, "w") as file:
        file.write(img_json)    



def main():

    # import json
    with open(full_file, 'r') as f:
        data = json.load(f)

    #print(json.dumps(data["images"], indent = 4, sort_keys=True))

#    for image in data["images"]:
#        print(json.dumps(image, indent = 4, sort_keys=True))
    start = time.time()
    Parallel(n_jobs=4)(delayed(create_img_json)(img_info) for img_info in data["images"])  
    stop = time.time()
    duration = stop - start
    print("Parallel:" + str(duration))

    # start = time.time()
    # for img_info in data["images"]:
    #     create_img_json(img_info)
    # stop = time.time()
    # duration = stop - start
    # print("For:" + str(duration))

    return

    img_json = data["images"][1]
    img_png = img_json["file_name"].split(".")[0] + ".png"
    img_location = img_folder + img_png
#    img = np.array(Image.open(img_location))    

    img = cv2.imread(img_location)


    all_rgb_codes = img.reshape(-1, img.shape[-1])

    unique_rgbs = np.unique(all_rgb_codes, axis=0)

    print("Unique:")
    print(str(unique_rgbs))

    # cv2.imshow("image!", img, img_bgr)

    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 

    # get the segments entry where the file_name is the img_png
    segments_json = next(item for item in data["annotations"] if item["file_name"] == img_png)
    print(segments_json)
    for segment in segments_json["segments_info"]:
        rle = []
        print(segment)
        #hex_id = format(segment["id"],"x")
        rgb_id = list(ImageColor.getcolor("#"+format(segment["id"],"x"),"RGB"))
        print(rgb_id)
        mask = np.all(img==rgb_id, axis=2)
        print(mask)
        height,width = mask.shape
        print(height)
        print()

        flat_mask = mask.flatten()

        # get points where mask[x] =! mask[x+1]
        # [0,0,0,1,1,0] -> [False, False,  True, False,  True]
        diff_mask = flat_mask[1:] != flat_mask[:-1]
        # get positions of these points
        # [False, False,  True, False,  True] -> [2,4]
        positions = np.where(diff_mask)
        # starting from -1, use np diff to get the lengths of each segment by
        # subtracting each position
        # first, expanding [2,4] to [-1,2,4,5]
        # [-1,2,4,5] -> [3,2,1]
        counts = np.diff(np.append(-1,positions,len(flat_mask)-1))
        print(counts)



    # coco notes that ids=R+G*256+B*256^2


    # ids=R+G*256+B*256^2
    #  [ 58 132 101]
    #  [ 90 128 144]]

    # to rgb: [ 101 132 58  ]

    # 58 + (132*256) + (101 * 256 *256)




 #   print(json.dumps(segments_json, indent = 4, sort_keys=True))

if __name__ == "__main__":
    main()
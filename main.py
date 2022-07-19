import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageColor
from joblib import Parallel, delayed
import time

img_folder = "panoptic_val2017/"
json_folder = "image_json/" # folder for per-image json outputs
short_file = "short_file.json" # an edited version of panoptic_val2017.json for testing
full_file = "panoptic_val2017.json"

#img_info is from the "images" list in the original json
def create_img_json(img_info,annotations,categories):
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

    img = cv2.imread(img_location)
    segments_json = next(item for item in annotations if item["file_name"] == imp_name_png)
    for segment in segments_json["segments_info"]:
        #print(segment)
        #hex_id = format(segment["id"],"x")
        # convert the id into RGB
        # per the COCO website, ids=R+G*256+B*256^2,
        # so the ids can be read as hex RGB color codes
        rgb_id = list(ImageColor.getcolor("#"+'{:06X}'.format(segment["id"]),"RGB"))
        mask = np.all(img==rgb_id, axis=2)
        height,width = mask.shape

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


        category = next(item for item in categories if item["id"] == segment["category_id"])

        segment_info = {
            "id": segment["id"],
            "category_name": category["name"],
            "mask": {
                "counts": counts.tolist(),
                "width": width,
                "height": height
            }
        }

        img_dict["segments_info"].append(segment_info)
        #print(counts)

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
    Parallel(n_jobs=12)(delayed(create_img_json)(img_info,data["annotations"],data["categories"]) for img_info in data["images"])  
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





 #   print(json.dumps(segments_json, indent = 4, sort_keys=True))

if __name__ == "__main__":
    main()
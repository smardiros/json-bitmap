import json
import cv2

img_folder = "panoptic_val2017/"

def main():

    # import json
    with open('short_file.json', 'r') as f:
        data = json.load(f)

    #print(json.dumps(data["images"], indent = 4, sort_keys=True))

    for image in data["images"]:
        print(json.dumps(image, indent = 4, sort_keys=True))

    img_json = data["images"][2]
    img_location = img_folder + img_json["file_name"].split(".")[0] + ".png"

    img = cv2.imread(img_location)

    cv2.imshow("image!", img)

    cv2.waitKey(0) 
    cv2.destroyAllWindows() 



if __name__ == "__main__":
    main()
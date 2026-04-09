import cv2
import matplotlib.pyplot as plt
import pandas as pd
import sys

def visualize_image(df, image_number):
    """
    Visualizes the nodule by using the x,y positions of the center of the nodule
    and draws a circle around the nodule based on the diameter of said nodulee

    ?Args: 
        df (pandas dataframe): dataframe containing meta data about the nodules
        image_number (int): number indicating the specific CXR you would like to look at

    ?Returns:
        creates a copy of original image with indicated nodule and saves it to 
        /data_sets/JSRT/highlighted_nodules
    """

    # Extract nodule info
    row = df.iloc[image_number - 1]
    image_name = "data_sets/JSRT/images\images/" + row["study_id"]
    print(image_name)
    image = cv2.imread(image_name)
    x = int(row["x"])
    y = int(row["y"])
    diameter = int(row["size"])
    transformed_diameter = transform_mm_to_pixels(diameter)


    # Circle config
    color = (0, 0, 255)
    thickness = 3

    # Draw circle around nodule
    cv2.circle(image, (x, y), transformed_diameter, color, thickness)

    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Save image
    path = "/data_sets/JSRT/highlighed_nodules/annotaded_" + image_name
    #cv2.imwrite(path, image) 

def read_data_from_file(file_path = "data_sets/JSRT/jsrt_metadata.csv"):
    return pd.read_csv(file_path)

def transform_mm_to_pixels(diameter):
    """
    Nodule size is given in milimeters(mm), convert the nodule size in mm to the approximate pixel size (px)

    ?Args: 
        diameter (int): diameter of a nodule in mm

    * DPI: Standard screen resolution of 96 DPI, 1mm is approx 3.78 pixels
    * INCH_IN_MM: 25.4mm = 1inch = 96 pixels
    
    ?Return:
        transformed_diameter (int) =  of nodule in pixel
    """
    DPI = 96
    INCH_IN_MM = 25.4

    return int((diameter * DPI) / INCH_IN_MM)

if __name__ == "__main__":
    image_number = int(input("Enter CXR image number: "))

    if image_number < 1 or image_number > 248:
        sys.exit("Error: Please enter a number in the range [1, 248]")
    
    df = read_data_from_file()
    visualize_image(df, image_number)




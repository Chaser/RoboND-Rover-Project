## Project: Search and Sample Return
## Engineer: Chase Johnson

---

**Aim:**  The aim of the `Search and Sample Return` project is to give experience of all three essential elements of robotics - `Perception`, `decision making`, and `action`. This will be achieved by receiving raw image data from a "rover" to navigate and avoid obstacticles while locating rocks.

This project is inspired by the [NASA sample return challenge](https://www.nasa.gov/directorates/spacetech/centennial_challenges/sample_return_robot/index.html)


### Notebook Analysis

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg
[image4]: ./report_data/perspective_transform_example.jpg
[image5]: ./report_data/navigable_color_threshed_example.jpg
[image6]: ./report_data/obstacle_color_threshed_example.jpg
[image7]: ./report_data/rock_color_threshold_example.jpg
[image8]: ./report_data/map_transform_example.png
[image9]: ./report_data/world_map_transform_example.png
[image10]: ./report_data/process_image_result.png

#### Perception through image analysis

The `perception` element of the project consists of following main steps:

1. Perspective Transform
2. Color Transforms
3. Coordinate Transforms

**1. Perspective Transform**

The perspective transform input is a raw image from the forward facing camera (image below) and `warps` the perspective to be a defined area within the output image, creating a top-down view.

Foward facing rover camera.

![alt text][image2]

Perspective transform output based on the input image and warping based on destination grid parameters.

![alt text][image4]

To achieve this the function `perspect_transform` transform takes three parameters the `img`, the `src` grid co-ordinates and the `dst` co-ordinates to return the warped perspective.

```python
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped
```

**2. Color Transform**

Determination of naviagable terrain, obstacles and rocks by isolating different color isolation and differenation of terrain objects.

**Navigable Terrain**

The `color_thresh` function evaluates `rgb_thresh` parameter against the `img` parameter for each pixel (x,y). This results in a binary image where 1 means the RGB was meet for the given `rgb_thresh`.

```python
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
```

The default and recommended `R=160,G=160,B=160` performed well for determining `nagivable` terrain.

![alt text][image4]

![alt text][image5]

**Obstacles**

As obstacle terrain can be considered the `inverse` of navigable the following code was used to retrieve the obstracle areas in the image

```python
obstacle_threshold = np.absolute(np.float32(threshed) - 1)
```

This resulted in the following result

![alt text][image4]

![alt text][image6]

**Rock Samples**

To identify rock samples a new function was required as it required above levels of `R=100` and `G=100` and less than `B=60`.

```python
def rock_thresh(img, rgb_thresh=(100, 100, 60)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    thresh = (img[:,:,0] > rgb_thresh[0]) \
                    & (img[:,:,1] > rgb_thresh[1]) \
                    & (img[:,:,2] < rgb_thresh[2])

    # Index the array of zeros with the boolean array and set to 1
    color_select[thresh] = 1
    # Return the binary image
    return color_select
```

The following was the result of the `rock_thresh` process

![alt text][image3]

![alt text][image7]

**3. Coordinate Transforms**

Up until now we have been dealing with a specific image in space/time co-ordinates. As this particular instance is required to become part of a `set` to aid in the overall decision making process another process needs to be applied. This process takes the captured rover image eand applies it to the overal known map, known as the `world map`.

![alt text][image8]

To achieve this the image co-ordinates are converted to rover coordinates using a function `rover_coords`

```python
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel
```

The rover coordinates are then used as input arguments to `pix_to_world` which rotates and translates the rover coordinate image.

```python
# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world
```

An example of such transform can be seen below.

![alt text][image9]

Once the captured image is transformed its important to store this data on the world map

#### Mapping

The image analysis steps can be combined and then utlized to provide a complete `perception` model to aid in the overall decision making. This was done in the `process_image(img)` function which consists of 7 discrete steps.

Step 1, consists of defining the source and destination points for the perspective transform.

```python
dst_size = 5
rover_fov_offset = 5
destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - rover_fov_offset],
                [img.shape[1]/2 + dst_size, img.shape[0] - rover_fov_offset],
                [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - rover_fov_offset], 
                [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - rover_fov_offset],
                ])
```

Next (Step 2), the perspective transform is applied to the image to get the top-down view.

```python
warped = perspect_transform(img, source, destination)
```

With the top-down view available from the `perspective_transform` the color threshold is applied (Step 3) to identify naviable terrain, obstacles and rock samples.

```python
navigable_threshold = color_thresh(warped)
obstacle_threshold = np.absolute(np.float32(navigable_threshold) - 1)
rock_threshold = rock_thresh(warped)
```

The image co-ordinates are then converted to rover coordinates using a function `rover_coords`

```python
navigable_xpix, navigable_ypix = rover_coords(navigable_threshold)
obstacle_xpix, obstacle_ypix = rover_coords(obstacle_threshold)
rock_xpix, rock_ypix = rover_coords(rock_threshold)
```

With the rover coordinates now available step 5 converts these values to world coords

```python
 # Acquire Navigable world data
navigable_x_world, navigable_y_world = pix_to_world(navigable_xpix, navigable_ypix, rover_xpos, rover_ypos,rover_yaw, data.worldmap.shape[0], scale)

# Acquire Obstacle world data
obstacle_x_world, obstacle_y_world = pix_to_world(obstacle_xpix, obstacle_ypix, rover_xpos, rover_ypos, rover_yaw,data.worldmap.shape[0], scale)

# Acquire Rock world data
rock_x_world, rock_y_world = pix_to_world(navigable_xpix, navigable_ypix, rover_xpos, rover_ypos, rover_yaw,data.worldmap.shape[0], scale)
```

With world-cordinates known for the particular image the image can be added to the `set` or bucket of current images captured and process. This is done in Step 6 below.

```python
data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 255
data.worldmap[rock_y_world, rock_x_world, 1] += 255  
data.worldmap[navigable_y_world, navigable_x_world, 2] += 255
```

An example of the **complete** `process_image` process can be seen below.
![alt text][image10]






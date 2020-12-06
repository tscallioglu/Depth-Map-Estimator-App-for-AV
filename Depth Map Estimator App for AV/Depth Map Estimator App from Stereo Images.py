import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches


import files_management




# Read the stereo-pair of images.
img_left = files_management.read_left_image()
img_right = files_management.read_right_image()

# Displays the two images.
_, image_cells = plt.subplots(1, 2, figsize=(10, 10))
image_cells[0].imshow(img_left)
image_cells[0].set_title('Left Image')
image_cells[1].imshow(img_right)
image_cells[1].set_title('Right Image')
plt.show()




# Reads the calibration.
p_left, p_right = files_management.get_projection_matrices()

np.set_printoptions(suppress=True)

print("p_left \n", p_left)
print("\np_right \n", p_right)



### ESTIMATING DEPTH
def compute_left_disparity_map(img_left, img_right):

     # Parameters.
    num_disparities = 6*16
    block_size = 11
    
    min_disparity = 0
    window_size = 6
    
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    
    # Stereo BM matcher.
    left_matcher_BM = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )

    # Stereo SGBM matcher.
    left_matcher_SGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Computes the left disparity map.
    disp_left = left_matcher_SGBM.compute(img_left, img_right).astype(np.float32)/16   

    return disp_left



# Computes the disparity map using the fuction above.
disp_left = compute_left_disparity_map(img_left, img_right)

# Shows the left disparity map.
plt.figure(figsize=(10, 10))
plt.title("Left Disparity Map")
plt.imshow(disp_left)
plt.show()



def decompose_projection_matrix(p):

    k, r, t,  _, _, _, _  = cv2.decomposeProjectionMatrix( p)
    
    t = t / t[3]
    
    return k, r, t



# Decomposes each P matrix.
k_left, r_left, t_left = decompose_projection_matrix(p_left)
k_right, r_right, t_right = decompose_projection_matrix(p_right)

# Displays the matrices.
print("k_left \n", k_left)
print("\nr_left \n", r_left)
print("\nt_left \n", t_left)
print("\nk_right \n", k_right)
print("\nr_right \n", r_right)
print("\nt_right \n", t_right)




def calc_depth_map(disp_left, k_left, t_left, t_right):

    f=k_left[0,0]
    b=t_left[1]-t_right[1]

    # Replaces all instances of 0 and -1 disparity with a small minimum value. (to avoid div by 0 or negatives)
    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1
    
    # Initializes the depth map to match the size of the disparity map.
    depth_map = np.ones(disp_left.shape, np.single)

    # Calculate the depths.
    depth_map[:] = f * b / disp_left[:]
    
    print("\n Depth Map minimum Dsitance: " + str(depth_map.min()))
    
    return depth_map




# Gets the depth map by calling the above function.
depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)

# Displays the depth map.
plt.figure(figsize=(8, 8), dpi=100)
plt.title("Depth Map")
plt.imshow(depth_map_left, cmap='flag')
plt.show()


### FINDING THE DISTANCE TO COLLISION
# Gets the image of the obstacle.
obstacle_image = files_management.get_obstacle_image()

# Shows the obstacle image.
plt.figure(figsize=(4, 4))
plt.title("Obstacle Image")
plt.imshow(obstacle_image)
plt.show()


def locate_obstacle_in_image(image, obstacle_image):
    
    cross_corr_map = cv2.matchTemplate(image,obstacle_image, cv2.TM_SQDIFF_NORMED)
    _, _, obstacle_location, _ , = cv2.minMaxLoc(cross_corr_map)

    return cross_corr_map, obstacle_location



# Gathers the cross correlation map and the obstacle location in the image.
cross_corr_map, obstacle_location = locate_obstacle_in_image(img_left, obstacle_image)

# Displays the cross correlation heatmap. 
plt.figure(figsize=(10, 10))
plt.title("The Cross Correlation Heatmap")
plt.imshow(cross_corr_map)
plt.show()

# Prints the obstacle location
print("\n Obstacle_location \n", obstacle_location)


def calculate_nearest_point(depth_map, obstacle_location, obstacle_img):

    obstacle_width=obstacle_img.shape[0]
    obstacle_height=obstacle_img.shape[1]
    
    obstacle_min_x_pos= obstacle_location [1]
    obstacle_min_y_pos= obstacle_location [0]
    
    
    obstacle_max_x_pos=obstacle_min_x_pos+obstacle_width
    obstacle_max_y_pos= obstacle_min_y_pos + obstacle_height
    
    obstacle_depth= depth_map_left[obstacle_min_x_pos:obstacle_max_x_pos, obstacle_min_y_pos: obstacle_max_y_pos]
    closest_point_depth=obstacle_depth.min()

    
    # Creates the obstacle bounding box.
    obstacle_bbox = patches.Rectangle((obstacle_min_y_pos, obstacle_min_x_pos), obstacle_height, obstacle_width,
                                 linewidth=1, edgecolor='r', facecolor='none')
    
    return closest_point_depth, obstacle_bbox




# Gets the closest point depth and obstacle bounding box.
closest_point_depth, obstacle_bbox = calculate_nearest_point(depth_map_left, obstacle_location, obstacle_image)




print("\n \n \n SUMMARY")

# Displays the image with the bounding box displayed.
fig, ax = plt.subplots(1, figsize=(10, 10))
plt.title("Image with Obstacle Image Bounding Box")
ax.imshow(img_left)
ax.add_patch(obstacle_bbox)
plt.show()

# Prints the depth of the nearest point.
print("closest_point_depth {0:0.3f}".format(closest_point_depth))



# Part 1. Read Input Data
img_left = files_management.read_left_image()
img_right = files_management.read_right_image()
p_left, p_right = files_management.get_projection_matrices()


# Part 2. Estimating Depth
disp_left = compute_left_disparity_map(img_left, img_right)
k_left, r_left, t_left = decompose_projection_matrix(p_left)
k_right, r_right, t_right = decompose_projection_matrix(p_right)
depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)


# Part 3. Finding the distance to collision
obstacle_image = files_management.get_obstacle_image()
cross_corr_map, obstacle_location = locate_obstacle_in_image(img_left, obstacle_image)
closest_point_depth, obstacle_bbox = calculate_nearest_point(depth_map_left, obstacle_location, obstacle_image)


# Print Result Output
print("Left Projection Matrix Decomposition:\n {0}".format([k_left.tolist(), 
                                                            r_left.tolist(), 
                                                            t_left.tolist()]))
print("\nRight Projection Matrix Decomposition:\n {0}".format([k_right.tolist(), 
                                                               r_right.tolist(), 
                                                               t_right.tolist()]))
print("\nObstacle Location (left-top corner coordinates):\n {0}".format(list(obstacle_location)))
print("\nClosest point depth (meters):\n {0}".format(closest_point_depth))



# Global boolean variables
USE_CUPY = False  # Set to True to use CuPy, False to use NumPy, NumPy seems to be faster
USE_WHITE_MODE = True  # Set to True to enable white mode functionality, this will make the lights automatically switch to white mode to get a more natural white
USE_DOMINANT_COLOR_IN_WHITE_MODE = False  # Set to True to use find_dominant_color in white mode, very slow and doesnt really improve the result
TimeDebug = False # execution duration of some functions for debugging
FPSEnabled = False # shows the speed in which the for loop processes the image
target_hz = 45  # Target frequency in Hz
WhiteBrightnessModifier = 13 # Affects brightness when AllowFullDark is True, 13 is darkest and everything higher and lower than it seems to be brighter
OtherLightModifier = 4 # 2 is default but i like my lights to be dark when my screen is dark
MaxWhiteBrightness = 30 # Goes up to 100
AllowFullDark = False # Allows lights to go off in case of black screen


import os
# Set up devices using your configurations

DEVICEID = os.getenv("DEVICEID", "<ID Here>")
DEVICEIP = os.getenv("DEVICEIP", "<IP Here>")
DEVICEKEY = os.getenv("DEVICEKEY", "<Key Here>")
DEVICEVERS = os.getenv("DEVICEVERS", "3.3")

DEVICEID2 = os.getenv("DEVICEID2", "<ID Here>")
DEVICEIP2 = os.getenv("DEVICEIP2", "<IP Here>")
DEVICEKEY2 = os.getenv("DEVICEKEY2", "<Key Here>")
DEVICEVERS2 = os.getenv("DEVICEVERS", "3.3")



if USE_CUPY:
    import cupy as cp
import mss
import numpy as np
from PIL import Image
import tinytuya
import time
import psutil
import gc
import threading
from sklearn.cluster import KMeans





# Create device instances
device = tinytuya.BulbDevice(dev_id=DEVICEID, address=DEVICEIP, local_key=DEVICEKEY)
device2 = tinytuya.BulbDevice(dev_id=DEVICEID2, address=DEVICEIP2, local_key=DEVICEKEY2)

device.set_version(3.3)
device.set_socketPersistent(True)
device.set_socketNODELAY(True)
device.set_socketRetryLimit(1)
device.set_retry(retry=False)

device2.set_version(3.3)
device2.set_socketPersistent(True)
device2.set_socketNODELAY(True)
device2.set_socketRetryLimit(1)
device2.set_retry(retry=False)

print("TinyTuya [%s] - ScreenSync\nBy Raycast_" % tinytuya.__version__)
print('TESTING: Device %s at %s with key %s version %s' %
      (DEVICEID, DEVICEIP, DEVICEKEY, DEVICEVERS))
print('TESTING: Device %s at %s with key %s version %s' %
      (DEVICEID2, DEVICEIP2, DEVICEKEY2, DEVICEVERS2))

def set_high_priority(pid=None):
    pid = pid or os.getpid()
    p = psutil.Process(pid)
    p.nice(psutil.HIGH_PRIORITY_CLASS)

# Call the function to set high priority for the current script
set_high_priority()

def time_function(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        if TimeDebug:
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            print(f"Function '{func.__name__}' executed in {duration:.4f} seconds")
            return result
        result = func(*args, **kwargs)
        return result
    return wrapper

def average_screen_color(screenshot):
    """Calculate average color from a screenshot."""
    img_array = cp.array(np.array(screenshot)) if USE_CUPY else np.array(screenshot)
    avg_color = cp.mean(img_array, axis=(0, 1)).astype(int) if USE_CUPY else np.mean(img_array, axis=(0, 1)).astype(int)
    return tuple(cp.asnumpy(avg_color)) if USE_CUPY else tuple(avg_color)

def is_close_to_white(color, threshold=200):
    """Check if the RGB color is close to white."""
    r, g, b = color
    return r >= threshold and g >= threshold and b >= threshold

def is_dark_color(color, threshold=60):
    """Check if the RGB color is dark or grayish."""
    r, g, b = color
    brightness = (r + g + b) / 3
    return brightness < threshold

def rgb_to_kelvin(rgb):
    """Convert RGB values to an approximate color temperature in Kelvin."""
    r, g, b = rgb
    if max(rgb) == 0:  # Avoid division by zero
        return 2700  # Default to a warm white

    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    temp_kelvin = 1000 + (r_norm * 1000) + (g_norm * 2000) + (b_norm * 3000)
    return max(2700, min(int(temp_kelvin), 6500))  # Clamp between 2700K and 6500K

def rgb_to_brightness(rgb):
    """Calculate brightness as a percentage based on the maximum RGB value."""
    max_rgb = max(rgb)
    return int((max_rgb / 255) * 100)  # Convert to percentage

@time_function
def find_dominant_color(image, threshold=200, num_colors=5, resize_factor=0.01):
    """Find the most dominant color in the image using K-Means clustering."""
    new_size = (int(image.width * resize_factor), int(image.height * resize_factor))
    resized_image = image.resize(new_size, Image.LANCZOS)
    img_array = cp.array(np.array(resized_image)) if USE_CUPY else np.array(resized_image)

    img_array = img_array.reshape(-1, img_array.shape[-1])
    img_array = img_array[cp.sum(img_array, axis=1) >= threshold * 3] if USE_CUPY else img_array[np.sum(img_array, axis=1) >= threshold * 3]

    if img_array.size == 0:
        return (0, 0, 0)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10)
    kmeans.fit(cp.asnumpy(img_array)) if USE_CUPY else kmeans.fit(img_array)  # Use appropriate array

    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color_index = counts.argmax()
    dominant_color = kmeans.cluster_centers_[dominant_color_index].astype(int)

    return tuple(dominant_color)

def rgb_to_color_temp(rgb):
    """Calculate color temperature as a percentage (0-100) based on RGB values."""
    r, g, b = rgb
    if max(rgb) == 0:
        return 0  # Avoid division by zero, treat as cool white
    return max(0, min(int((r * 0.3 + g * 0.59 + b * 0.11) / 255 * 100), 100))

@time_function
def set_device_color(device, color):
    """Set the device color or adjust for white mode based on average RGB color."""
    if USE_WHITE_MODE and (is_close_to_white(color) or is_dark_color(color)):
        #kelvin_temp = rgb_to_kelvin(color)
        brightness = rgb_to_brightness(color)
        color_temp = rgb_to_color_temp(color)
        #print(is_dark_color(color))
        if is_dark_color(color) and not AllowFullDark:  # Check condition for brightness adjustment
            brightness = max(0, brightness // OtherLightModifier)  # Reduce brightness to avoid turning off completely
        elif brightness > WhiteBrightnessModifier:
            brightness = brightness - WhiteBrightnessModifier
        if brightness > MaxWhiteBrightness:
            brightness = MaxWhiteBrightness
        #if color_temp > 82:
        #    color_temp = 100
        #elif color_temp >= 15:
        #    color_temp = 0
        #print(color_temp)
        device.set_mode('white')  # Switch to white mode
        device.set_white_percentage(brightness, color_temp)  # Set to calculated brightness and temperature
    else:
        device.set_colour(color[0], color[1], color[2])  # Set RGB color

@time_function
def color_for_regions():
    """Capture and return colors from different regions of the screen."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        width, height = monitor["width"], monitor["height"]

        bottom_left_rect = {"top": height // 2, "left": 3, "width": width // 3, "height": height // 2}
        top_right_rect = {"top": 0, "left": width // 3 + width // 3, "width": width // 3, "height": height // 2}

        while True:
            bottom_left_screenshot = sct.grab(bottom_left_rect)
            top_right_screenshot = sct.grab(top_right_rect)

            # Check if the bottom left region is predominantly white
            bottom_left_image = Image.frombytes("RGB", bottom_left_screenshot.size, bottom_left_screenshot.bgra, "raw", "BGRX")
            bottom_left_avg_color = average_screen_color(bottom_left_image)

            if USE_WHITE_MODE:
                if USE_DOMINANT_COLOR_IN_WHITE_MODE and is_close_to_white(bottom_left_avg_color):
                    bottom_left_color = find_dominant_color(bottom_left_image)
                else:
                    bottom_left_color = bottom_left_avg_color
            else:
                bottom_left_color = bottom_left_avg_color

            # Check if the top right region is predominantly white
            top_right_image = Image.frombytes("RGB", top_right_screenshot.size, top_right_screenshot.bgra, "raw", "BGRX")
            top_right_avg_color = average_screen_color(top_right_image)

            if USE_WHITE_MODE:
                if USE_DOMINANT_COLOR_IN_WHITE_MODE and is_close_to_white(top_right_avg_color):
                    top_right_color = find_dominant_color(top_right_image)
                else:
                    top_right_color = top_right_avg_color
            else:
                top_right_color = top_right_avg_color

            yield bottom_left_color, top_right_color

# Usage example
gc.enable()

target_interval = 1 / target_hz
last_bottom_left_color, last_top_right_color = (0, 0, 0), (0, 0, 0)
color_generator = color_for_regions()

while True:
    start_time = time.time()

    bottom_left_color, top_right_color = next(color_generator)

    # Create threads for setting colors
    threads = []

    if bottom_left_color != last_bottom_left_color:
        thread1 = threading.Thread(target=set_device_color, args=(device, bottom_left_color))
        thread1.start()
        threads.append(thread1)
        last_bottom_left_color = bottom_left_color

    if top_right_color != last_top_right_color:
        thread2 = threading.Thread(target=set_device_color, args=(device2, top_right_color))
        thread2.start()
        threads.append(thread2)
        last_top_right_color = top_right_color

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    elapsed_time = time.time() - start_time
    sleep_time = max(0, target_interval - elapsed_time)

    if FPSEnabled:
        fps = 1 / elapsed_time if elapsed_time > 0 else 0  # Calculate frequency in Hz
        print(f"Iteration executed in {elapsed_time:.4f} seconds ({fps:.2f} FPS)")
        #print(sleep_time)

    time.sleep(sleep_time)

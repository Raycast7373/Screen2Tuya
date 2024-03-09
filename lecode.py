import os
import colorsys
import mss
import numpy as np
from PIL import Image
import tinytuya
import time
import psutil
import gc
def set_high_priority(pid=None):
    pid = pid or os.getpid()
    p = psutil.Process(pid)
    p.nice(psutil.HIGH_PRIORITY_CLASS)

# Call the function to set high priority for the current script
set_high_priority()
DEVICEID = "bf6aab8ebe79328335hpcg"
DEVICEIP = "192.168.1.94"
DEVICEKEY = "0d9e3f6b45165b06"
DEVICEVERS = "3.3"
refresh_rate = 1  # higher is slower

# Check for environmental variables and always use those if available
DEVICEID = os.getenv("DEVICEID", DEVICEID)
DEVICEIP = os.getenv("DEVICEIP", DEVICEIP)
DEVICEKEY = os.getenv("DEVICEKEY", DEVICEKEY)
DEVICEVERS = os.getenv("DEVICEVERS", DEVICEVERS)

print("TinyTuya - Smart Bulb RGB Test [%s]\n" % tinytuya.__version__)
print('TESTING: Device %s at %s with key %s version %s' %
      (DEVICEID, DEVICEIP, DEVICEKEY, DEVICEVERS))

device = tinytuya.BulbDevice(dev_id=DEVICEID, address=DEVICEIP, local_key=DEVICEKEY)
device.set_version(3.3)
device.set_socketPersistent(True)

def average_screen_color(screenshot):
    # Downscale the image to improve processing speed
    width, height = screenshot.size
    scaled_width, scaled_height = width // 20, height // 20  # Reduced downsampling size
    screenshot = screenshot.resize((scaled_width, scaled_height))

    # Convert the screenshot to numpy array
    img_array = np.array(screenshot)

    # Calculate the average color
    avg_color = np.mean(img_array, axis=(0, 1)).astype(int)

    return tuple(avg_color)

def increase_saturation(rgb_color, factor):
    # Convert RGB color to HSL color space
    h, l, s = colorsys.rgb_to_hls(rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)

    # Increase saturation by the specified factor
    s *= factor

    # Ensure saturation is within the valid range [0, 1]
    s = max(0, min(s, 1))

    # Convert back to RGB color space
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # Convert to integer RGB values
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)

    return (r, g, b)

def smooth_color_change(current_color, new_color, smoothing_factor):
    # Smooth out color changes using moving average
    smoothed_color = tuple(int((1 - smoothing_factor) * c1 + smoothing_factor * c2) for c1, c2 in zip(current_color, new_color))
    return smoothed_color

def color(factor, smoothing_factor):
    last_color = None  # Store the last color value

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Change index according to your monitor setup
        width, height = monitor["width"], monitor["height"]
        square_size = min(width, height) // 3  # Size of the square
        left = (width - square_size) // 3
        top = (height - square_size) // 3
        rect = left, top, left + square_size, top + square_size  # Define the square region

        while True:
            # Capture the screen
            screenshot = sct.grab(rect)

            # Calculate average color
            avg_color = average_screen_color(Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"))

            # Smooth out color changes
            #if last_color is not None:
            #    avg_color = smooth_color_change(last_color, avg_color, smoothing_factor)
            #last_color = avg_color

            # Increase saturation
            increased_saturation_color = increase_saturation(avg_color, factor)

            yield increased_saturation_color

# Example usage
saturation_factor = 3  # You can adjust this factor to increase/decrease saturation
smoothing_factor = 0.05  # You can adjust this factor for faster color transitions
last_color_thing = (0, 0, 0)
color_generator = color(saturation_factor, smoothing_factor)
gc.enable()
while True:
    # Get the next color from the generator
    average_color = next(color_generator)
    #print("Average Screen Color (RGB):", average_color)
    if not average_color == last_color_thing:
        #print("Ignored")
        device.set_colour(average_color[0], average_color[1], average_color[2])
        last_color_thing = average_color
    #print(average_color)
    #print(last_color_thing)
    #time.sleep(0.02)

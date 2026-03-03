# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import board
import digitalio
from PIL import Image, ImageDraw, ImageFont

import adafruit_ssd1306

# Change these as needed
WIDTH = 128
HEIGHT = 64  
BORDER = 5

# Use for I2C.
i2c = board.I2C()  # uses board.SCL and board.SDA
oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=0x3C)

# Clear display.
oled.fill(0)
oled.show()

# Create blank image for drawing.
# Make sure to create image with mode '1' for 1-bit color.
image = Image.new("1", (oled.width, oled.height))

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Draw a white background
draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)

# Load default font.
font = ImageFont.load_default(size = 14) ### https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
#font1 = ImageFont.load("arial.pil")
# Draw Some Text
text = "ECE 4415 Group 4"
bbox = font.getbbox(text)
'''(font_width, font_height) = bbox[2] - bbox[0], bbox[3] - bbox[1]
draw.text(
    (oled.width // 2 - font_width // 2, oled.height // 2 - font_height // 2),
    text,
    font=font,
    fill=255,
)
'''
draw.text(
    (0, 0),
    text,
    font=font,
    fill=255,
)
text1 = "Mode 1 Lane keep"
bbox = font.getbbox(text1)
draw.text(
    (0, 15),
    text1,
    font=font,
    fill=255,
)
text2 = "Throttle: xyz"
bbox = font.getbbox(text2)
draw.text(
    (0, 35),
    text2,
    font=font,
    fill=255,
)
text3 = "Steering: uvw"
bbox = font.getbbox(text3)
draw.text(
    (0, 45),
    text3,
    font=font,
    fill=255,
)

# Display image
oled.image(image)
oled.show()

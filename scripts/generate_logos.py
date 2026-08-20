#!/usr/bin/env python3
"""Generate OpenLawAI logo images programmatically using the Nunito font."""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "chatdb" / "static" / "chat"

# Font settings (matching the CSS: Nunito, bold, italic)
FONT_NAME = "Nunito"
TEXT = "OpenLawAI"
ICON_TEXT = "OL"

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def get_font(size: int, bold: bool = True, italic: bool = True) -> ImageFont.FreeTypeFont:
    """Get Nunito font with specified style, fallback to Ubuntu."""
    # Font paths to try in order of preference
    font_paths = []
    
    if bold and italic:
        font_paths = [
            "/usr/share/fonts/truetype/nunito/Nunito-BoldItalic.ttf",
            os.path.expanduser("~/.fonts/Nunito-BoldItalic.ttf"),
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-BI.ttf",  # Ubuntu Bold Italic
        ]
    elif bold:
        font_paths = [
            "/usr/share/fonts/truetype/nunito/Nunito-Bold.ttf",
            os.path.expanduser("~/.fonts/Nunito-Bold.ttf"),
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        ]
    
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    
    # Fallback: try Ubuntu variable font
    ubuntu_path = "/usr/share/fonts/truetype/ubuntu/Ubuntu-MI.ttf"
    if os.path.exists(ubuntu_path):
        print("Using Ubuntu font as fallback (install fonts-nunito for exact match)")
        return ImageFont.truetype(ubuntu_path, size)
    
    print("Warning: No suitable font found, using default")
    return ImageFont.load_default()


def generate_favicon(size: int, output_name: str):
    """Generate a square favicon with 'OL' text."""
    img = Image.new("RGBA", (size, size), WHITE)
    draw = ImageDraw.Draw(img)
    
    # Calculate font size (roughly 60% of image size)
    font_size = int(size * 0.55)
    font = get_font(font_size)
    
    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), ICON_TEXT, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (size - text_width) // 2 - bbox[0]
    y = (size - text_height) // 2 - bbox[1]
    
    draw.text((x, y), ICON_TEXT, font=font, fill=BLACK)
    
    output_path = OUTPUT_DIR / output_name
    img.save(output_path, "PNG")
    print(f"Generated: {output_path}")
    return img


def generate_ico(sizes: list[int], output_name: str):
    """Generate ICO file with multiple sizes."""
    images = []
    for size in sizes:
        img = Image.new("RGBA", (size, size), WHITE)
        draw = ImageDraw.Draw(img)
        
        font_size = int(size * 0.55)
        font = get_font(font_size)
        
        bbox = draw.textbbox((0, 0), ICON_TEXT, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size - text_width) // 2 - bbox[0]
        y = (size - text_height) // 2 - bbox[1]
        
        draw.text((x, y), ICON_TEXT, font=font, fill=BLACK)
        images.append(img)
    
    output_path = OUTPUT_DIR / output_name
    images[0].save(output_path, format="ICO", sizes=[(s, s) for s in sizes])
    print(f"Generated: {output_path}")


def generate_og_image(width: int = 1200, height: int = 630):
    """Generate Open Graph image for social media sharing."""
    img = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(img)
    
    # Main title - large
    font_size = int(height * 0.18)
    font = get_font(font_size)
    
    bbox = draw.textbbox((0, 0), TEXT, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2 - bbox[0]
    y = (height - text_height) // 2 - bbox[1]
    
    draw.text((x, y), TEXT, font=font, fill=BLACK)
    
    output_path = OUTPUT_DIR / "og-image.png"
    img.save(output_path, "PNG")
    print(f"Generated: {output_path}")


def generate_apple_touch_icon(size: int = 180):
    """Generate Apple touch icon."""
    img = Image.new("RGBA", (size, size), WHITE)
    draw = ImageDraw.Draw(img)
    
    font_size = int(size * 0.5)
    font = get_font(font_size)
    
    bbox = draw.textbbox((0, 0), ICON_TEXT, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size - text_width) // 2 - bbox[0]
    y = (size - text_height) // 2 - bbox[1]
    
    draw.text((x, y), ICON_TEXT, font=font, fill=BLACK)
    
    output_path = OUTPUT_DIR / "apple-touch-icon.png"
    img.save(output_path, "PNG")
    print(f"Generated: {output_path}")


def main():
    """Generate all logo images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Generating OpenLawAI logos...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Favicons
    generate_favicon(16, "favicon-16x16.png")
    generate_favicon(32, "favicon-32x32.png")
    generate_favicon(32, "favicon-32.png")
    generate_favicon(64, "favicon-64.png")
    
    # ICO file with multiple sizes
    generate_ico([16, 32, 48], "favicon.ico")
    
    # Apple touch icon
    generate_apple_touch_icon(180)
    
    # OG image for social sharing
    generate_og_image()
    
    print()
    print("Done! All logos generated.")


if __name__ == "__main__":
    main()

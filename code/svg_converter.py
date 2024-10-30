import os
import subprocess
from PIL import Image  # Pillow for image handling

# Function to convert SVG to PNG using Inkscape
def svg_to_png(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.svg'):
            svg_file = os.path.join(input_folder, filename)
            png_file = os.path.join(output_folder, filename.replace('.svg', '.png'))

            # Use Inkscape to convert SVG to PNG
            try:
                subprocess.run(
                    ['inkscape', svg_file, '--export-filename=' + png_file],
                    check=True
                )
                print(f"Converted {filename} to PNG.")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {filename}: {e}")
            except FileNotFoundError:
                print("Inkscape not found. Make sure it is installed and added to PATH.")


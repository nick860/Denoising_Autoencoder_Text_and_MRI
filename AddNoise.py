import os
from PIL import Image
import random
import numpy as np


def add_gaussian_noise(image, mean=0.2, var=0.15):
  """
  Adds Gaussian noise to a PIL image.

  Args:
      image: A PIL Image object.
      mean: Mean value of the Gaussian noise (default: 0.0).
      var: Variance of the Gaussian noise (default: 0.1).

  Returns:
      A new PIL Image object with Gaussian noise added.
  """
  width, height = image.size
  # Convert image to a NumPy array with the same shape as noise
  image_array = np.array(image) / 255
# make the image array in shape of (height, width)

  if len(image_array.shape) == 2:
     image_array = np.repeat(image_array[:, :, np.newaxis], 3, axis=2)

  noise = np.random.normal(mean, var, (height, width,3))
  noisy_image = (image_array+noise) * 255
  return Image.fromarray(noisy_image.astype(np.uint8)).convert("RGB")


def add_noise_to_directory(directory, output_directory1=None, output_directory2=None):
  """
  Adds Gaussian noise to all images in a directory and saves them optionally to a new directory.

  Args:
      directory: Path to the directory containing images.
      output_directory: Path to the directory for saving noisy images (optional, defaults to original directory).
  """
  if not output_directory1:
    output_directory1 = directory

  num_images = 0
  for filename in os.listdir(directory):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
      image_path = os.path.join(directory, filename)
      try:
        image = Image.open(image_path)
        input_filename = os.path.join(output_directory2, str(num_images)+".jpg")
        image.save(input_filename)
        noisy_image = add_gaussian_noise(image)
        output_filename = os.path.join(output_directory1, str(num_images)+".jpg")
        noisy_image.save(output_filename)
        num_images += 1
      except Exception as e:
        raise
        print(f"Error processing image: {image_path} - {e}")


if __name__ == "__main__":
  # Replace 'path/to/images' with the actual directory containing your images
  image_directory = os.path.join(os.getcwd(), 'dataset2\\image_add_noise') 
  output_directory1 = os.path.join(os.getcwd(), 'dataset2\\test')
  output_directory2 = os.path.join(os.getcwd(), 'dataset2\\dem')
  # Optional: Specify a different directory to save noisy images (default: overwrite originals)
  # noisy_image_directory = "path/to/noisy/images"
  add_noise_to_directory(image_directory,output_directory1, output_directory2)
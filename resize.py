#Importing Libraries
import argparse
import os,sys
from PIL import Image

#Implementing Classes and Methods
class Resizer(object):

	@classmethod
	def resize_image(self,image, size):
	
		#Modify  an image's dimension to propersize.
		return image.resize(size, Image.ANTIALIAS)

	@classmethod
	def resize_images(self,image_dir, output_dir, size):

		#Resize the Multiple image from directory 'image_dir' and save to 'output_dir'.
		if not os.path.exists(output_dir):
		os.makedirs(output_dir)

		
		images = os.listdir(image_dir)
		num_images = len(images)
		for i, image in enumerate(images):
			with open(os.path.join(image_dir, image), "r+b") as f:
				with Image.open(f) as img:
					img = resize_image(img, size)
					img.save(os.path.join(output_dir, image), img.format)

			if ((i+1) % 100) == 0:
				print ("[{}/{}] Resized the images and saved into '{}'".format(i+1, num_images, output_dir))

	@classmethod
	def main(self,args):

		image_dir = args.image_dir
		output_dir = args.output_dir
		image_size = [args.image_size, args.image_size]
		resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--image_dir', type=str, default='./data/train2017/', help="directory for train images")
	parser.add_argument('--output_dir', type=str, default='./data/resized2017/', help="directory for saving resized images")
	parser.add_argument('--image_size', type=int, default=256, help="size for image after processing")

	args = parser.parse_args()

	#creating objects
	obj = Resizer()
	obj.main(args)



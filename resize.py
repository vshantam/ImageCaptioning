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
					img = self.resize_image(img, size)
					img.save(os.path.join(output_dir, image), img.format)

			if ((i+1) % 100) == 0:
				print ("[{}/{}] Resized the images and saved into '{}'".format(i+1, num_images, output_dir))

	@classmethod
	def main(self,args):

		image_dir = './data/train2014/'
		output_dir = './data/resized2014/'
		image_size = [256, 256]
		self.resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':

	#creating objects
	obj = Resizer()
	obj.main()



import cv2
import numpy as np
from matplotlib import pyplot as plt

class opticalFlow:
	def __init__(self, image1, image2, blur_f = (10, 10)):
		self.image1 = image1
		self.image2 = image2
		self.blur_f = blur_f
		self.kernal_size = 10
		self.ksobel = 5
		self.width = 0
		self.height = 0

		self.x = []
		self.y = []
		self.u = []
		self.v = []

		#Sobel x and sobel y gets the gradient along x and y axis.
		processed_1 = self.preprocessing(image1)
		processed_2 = self.preprocessing(image2)

		#Both of the image should have the same height and width
		self.width, self.height = processed_1.shape
		#add padding to the image
		processed_1 = cv2.copyMakeBorder(processed_1,0,self.kernal_size,0,self.kernal_size,cv2.BORDER_REPLICATE)
		processed_2 = cv2.copyMakeBorder(processed_2,0,self.kernal_size,0,self.kernal_size,cv2.BORDER_REPLICATE)

		#Sober image for image
		sobelx_1 = cv2.Sobel(processed_1,cv2.CV_64F,1,0,ksize=self.ksobel)
		sobely_1 = cv2.Sobel(processed_1,cv2.CV_64F,0,1,ksize=self.ksobel)

		sobelx_2 = cv2.Sobel(processed_2,cv2.CV_64F,1,0,ksize=self.ksobel)
		sobely_2 = cv2.Sobel(processed_2,cv2.CV_64F,0,1,ksize=self.ksobel)

		#gradiant difference based on time as a variable is really just addition of x and y
		laplacian_1 = sobelx_1 + sobely_1
		laplacian_2 = sobelx_2 + sobely_2
		It = laplacian_2 - laplacian_1

		# cv2.imwrite('image1x.png', sobelx_1)
		# cv2.imwrite('image1y.png', sobely_1)

		# cv2.imwrite('image2x.png', sobelx_2)
		# cv2.imwrite('image2y.png', sobely_2)

		# cv2.imwrite("image1all.png", laplacian_1)
		# cv2.imwrite("image2all.png", laplacian_2)

		#Now loop through the whole image using the kernal defined(Should be parallelized)
		for i in range(0, self.width, self.kernal_size):
			for j in range(0, self.height, self.kernal_size):
				#For every kernal sized square, calculate the optical flow
				self.x.append(i)
				self.y.append(j)
				self.calculate(i, j, sobelx_1, sobely_1, It)

		#Let's see the plot
		self.plot()

	def plot(self):

		#Shrink the array
		for _ in range(10):
			for i in range(0, len(self.x), 2):
				self.x[i] = 0
				self.y[i] = 0
				self.u[i] = 0
				self.v[i] = 0

		fig, ax = plt.subplots()
		plt.imread("TestSeq/Shift0.png")
		X, Y = np.meshgrid(self.x, self.y)
		U, V = np.meshgrid(self.u, self.v)

		ax.quiver(X, Y, U, V, scale = 8, scale_units='inches')

		fig.savefig('test.png')


	#Using sobelx to calculate
	def calculate(self, x, y, sobelx, sobely, It):
		xx,xy,yy,xt,yt = 0,0,0,0,0
		for i in range(x, x+self.kernal_size):
			for j in range(y, y+self.kernal_size):
				xx += sobelx[i][j]*sobelx[i][j]
				xy += sobelx[i][j]*sobely[i][j]
				yy += sobely[i][j]*sobely[i][j]
				xt += sobelx[i][j]*It[i][j]
				yt += sobely[i][j]*It[i][j]
		ATA = [[xx,xy],[xy,yy]]
		ATB = [[xt],[yt]]
		#Now we got the displacement, be careful of singular matrix
		try:
			result = np.dot(np.linalg.inv(ATA), np.array(ATB)*-1)
		except np.linalg.LinAlgError:
			return 0
		self.u.append(result[0][0])
		self.v.append(result[1][0])

		# print(self.u)
		# print(self.v)
		return result

	def preprocessing(self, img):
		#Convert the imported frame to grey image
		#Blue the grey image
		gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blurImg = cv2.blur(gray_image,(self.blur_f[0], self.blur_f[1]))
		return blurImg

if __name__ == '__main__':
	#Later put into the user interface
	image1 = cv2.imread('TestSeq/Shift0.png')
	image2 = cv2.imread('TestSeq/ShiftR5U5.png')
	opticalFlow(image1, image2)
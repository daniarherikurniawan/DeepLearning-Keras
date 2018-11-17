Assignment 2

Imagmagick
- pick two framework
- search cats and dogs kaggle
- pull data using google image download
- measure the performance and inference for two different model on two different framework on 3 different platform
- Write up! detail model structure... 

- use google collab to specify cpu, tpu and gpu

tensorflow of poets/flower

- folder based tagging

measuring the execution performance
	preference recall

time for inference
	
bencmark stream

- TACO


==============================================================================================================
I pick : Keras/Tensorflow, Keras/Pytorch
			VGG, Resnet

https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/

from imagesoup import ImageSoup

soup = ImageSoup()
images_wanted = 1000
query = 'cheetah'
images = soup.search(query, n_images=images_wanted)

f = open('img-all/cheetah-url','w')
urls=''
for x in range(0,1000):
	index=x-n
	print(str(x) + " - " + str(index))
	urls=urls+str(images[x])+"\n"

f.write(urls)
f.close()


===============================================
var script = document.createElement('script');
script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(script);

var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });

var textToSave = urls.toArray().join('\n');
var hiddenElement = document.createElement('a');
hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
hiddenElement.target = '_blank';
hiddenElement.download = 'urls.txt';
hiddenElement.click();


sudo find /Users/daniar/Documents/Project/DL-Kaggle/dog-project/ -name ".DS_Store" -depth -exec rm {} \;

===============================================
from imutils import paths
import argparse
import requests
import cv2
import os

x=0
lines = [line.rstrip('\n') for line in open('img-all/tiger-url')]


for url in lines:
	url = url.strip()
	print(url)
	try:
		r = requests.get(url, timeout=5000)
		# save the image to disk
		f = open("img-all/tiger/tiger."+str(x)+".jpg", "wb")
		f.write(r.content)
		f.close()
		x=x+1
	except:
		print("[INFO] error downloading... " + str(x))












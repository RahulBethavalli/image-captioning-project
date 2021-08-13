# image-captioning-project
In this project, we want to generate automatic image captions from a given image. In other words, caption what is going on in the image. We used [Flickr8k](https://www.kaggle.com/srbhshinde/flickr8k-sau) dataset including 8000+ images and 5 reference sentences provided by human annotators.

The task can be divided into two modules – one is an image based model – which extracts the features and nuances out of our image, and the other is a language based model – which translates the features and objects given by our image based model to a natural sentence.
![alt text](https://github.com/RahulBethavalli/image-captioning-project/blob/main/png/model.png)

We used a convolutional neural network like the VGG network. Then we sent the image through this CNN and we removed the last softmax layer. The network used here is a pretrained model [Inception V3]("https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth")and we just fine-tuned the last layer, which will then act as an input to the LSTM.

# Captions predicted by the model
![few epochs](https://github.com/RahulBethavalli/image-captioning-project/blob/main/png/result1.png)
![100 epochs](https://github.com/RahulBethavalli/image-captioning-project/blob/main/png/result2.png)

We also implemented attention and validated the caption predicted by the trained model using Bilingual evaluation understudy (BLEU), which is a well-acknowledged metric to measure the similarly of one hypothesis sentence to multiple reference sentences. Given a single hypothesis sentence and multiple reference sentences, it returns value between 0 and 1. The metric close to 1 means that the two are very similar.

The prediction is not very accurate but fairly good considering the size of the dataset.
[Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/) or [MS-COCO](http://cocodataset.org/#download) can be used to more accurate captions.

# Summary
1.Our model depends on the data, so it can't predict the words that are out of its vocabulary.
2.We used a small dataset consisting of 8000 images. For production-level models, we need to train on datasets larger than 100,000 images which can produce better accuracy models.
3.We saw the improvements resulted in the model by adding attention.

# Future Improvements
1.We are going to implement beam search and compare the results with the greedy approach
2.We will compare the difference in performance using global and local attentions

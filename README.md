# tensorflow_like_caffe
## 建立初衷
tensorflow比caffe入门要难许多，但是感觉比caffe可操作性强。这份代码是为了以后训练网络不要重复进行tensorflow多GPU并行、数据增强（翻转、旋转、裁剪等）等操作而写的。
## 参考源码
我阅读了tf的[官网关于多GPU实现](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py)的部分，以及[facenet中多线程](https://github.com/davidsandberg/facenet/blob/096ed770f163957c1e56efa7feeb194773920f6e/src/train_softmax.py#L110)读取数据+预处理（数据增强、shuffle）的部分，还有一个实现多GPU的VGG的[github](https://github.com/huyng/tensorflow-vgg/blob/master/train_model_parallel.py)。
## 以后的改进
数据流水线似乎还不是太好，很难有多块GPU长时间维持在80%~100%的利用率。
## 代码内容（tensorflow_like_caffe）
1、align文件夹——dlib 68 点人脸检测+对齐（对齐可选）<br>
2、experiment.yaml——类似于caffe的solver配置文件<br>
3、layers.py——定义一些tensor<br>
4、tools.py——加载.npz/.npy文件（把caffe model转到tf时会用到）<br>
5、train_data_generator.py（流水线读取数据）<br>
6、train_model_parallel.py（主文件）<br>
7、vgg.py<br>
## 相关信息
最近在实现[VGG face](http://202.116.81.74/cache/16/03/www.robots.ox.ac.uk/7ae360b2340988ebb009f723db77afc7/parkhi15.pdf)的triplet loss，所以才从caffe转向tf，等复现了论文再来更新一波。

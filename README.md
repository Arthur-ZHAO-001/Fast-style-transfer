# 风格迁移小程序的开发与部署

## 小组成员：赵呈亮 孙易泽 吴锦程 许函嘉  詹沛

## 1.图片风格迁移原理


**风格迁移(neural style transfer)**，是一种基于卷积神经网络来生成图片算法。我们选取一张风格图片（style）与一张内容图片（content），算法会自动生成一张新的图片。具体效果如下图所示；我们选取内容图片为山东大学（威海）校门的照片，风格图片选取梵高著名的《星月夜》。训练好的算法会自动生成一张带有《星月夜》风格的校门照片。风格迁移技术完美的将深度学习技术应用于艺术领域，兼具了科技与艺术的美感。
<img src="https://s1.ax1x.com/2020/10/05/0tMhp8.png" alt="0tMhp8.png" border="0" width=80%/>

### 1.1 损失函数定义
为了构建一个神经风格转移系统 我们为生成的图像来定义一个代价函数,通过最小化这个代价函数来生成最后的图片, 在Gatys的论文中，它定义整体损失函数如下

$$ J(G)=\alpha J_{content}(C,G)+\beta J_{style}(S,G)$$



其中，$J_{content}$衡量生成图片与内容图片在内容上的差距； $J_{style}$表示生成图片与风格图片在风格、纹理上的差别。定义完损失函数之后，我们可以用梯度下降的方式不断最小化这个损失函数。 



通过这个这个过程中，我们实际上是在更新生成图片中每一个像素点。

* **内容损失(content loss)**


在卷积神经网络中，每一层神经网络都会利用上一层的输出来进一步提取更加复杂的特征，直到复杂到能被用来识别物体为止，所以每一层都可以被看做很多个局部特征的提取器。我们想利用这些已经被训练好的神经网络来衡量生成图片与内容图片间的相似程度。一般认为较低层的描述了图像的具体视觉特征（即纹理、颜色等），较高层的特征是较为抽象的图像内容描述。所以我们比较两幅图片网络中高层特征的相似性即可。具体我们提取出图片经过神经网络的特定层运算出的结果，即为此特定层的激活因子。当两张图片的的激活因子定义内容损失为

$$J_{content}=\frac{1}{2}\left \| a^{[l](C)} - a^{[l](G)}\right \|$$

其中$ a^{[l](C)}$与$ a^{[l](G)}$ 分别表示内容图片和生成图片在神经网络的第l曾的激活因子（在CNN中为矩阵形式）。在此，一般选取VGG16或VGG19作为提取特征的预训练网络

* **风格损失(style loss)**
**风格**在艺术上是一种很抽象的概念，但往往某些画中的形状都有特定色彩。以《星月夜》为例,圆圈形状总是蓝色或黄色的。所以如果在一张图片中有很多蓝色或黄色的同心圆，我们可以认为它具有《星月夜》的某些特征。而卷积神经网络的卷积层即为特征提取器，不同通道的卷积核可能提取不同的图像特征；如某一特定颜色或形状。如果计算这些特征之间的相关性，如果相关性较大，就说明这张看起来更具这种风格。所以，我们使用预训练网络VGG16来计算风格损失。

    **gram矩阵**：
$$G_{k,k{}'}^{[l]}=\sum_{i=1}^{n_{H}^{[l]}}\sum_{i=1}^{n_{W}^{[l]}}a_{i,j,k}^{[l]}a_{i,j,k{}'}^{[l]}$$

其中 $a_{i,j,n}^{[l]}$为图片经过网络第l层计算后，第n个通道，第i行j列的激活值。 gram矩阵$G^{[l]}$ 是一个$n_{H}^{[l]} \times  n_{W}^{[l]}$ 的矩阵。 gram矩阵的主要作用为衡量在神经网络的特定层 不同通道间的相关性。其本质为每一层卷积核计算后生成的的激活因子矩阵（feature map）, 后将其转置并相乘得到的矩阵。

在定义了gram矩阵后，我们可以用网络中特定层的gram矩阵来衡量生成图片与风格图片。定义风格损失如下

$$J_{style}^{[l]}=\frac{1}{(2n_{H}^{[l]}n_{W}^{[l]}n_{C}^{[l]})}\sum_{k}^{}\sum_{k{}'}^{}(G_{k,k{}'}^{[l](G)}-G_{k,k{}'}^{[l](S)})^2$$





### 1.2 原始风格迁移[<sup>1</sup>](#refer-anchor-1)
<img src="https://s1.ax1x.com/2020/10/06/0Njtit.png" alt="0Njtit.png" border="0"  width=70%/>
在Gatys的论文中，它使用了VGG19 网络作为内容风格的提取网络。其中风格损失选取Conv1_2、Conv2_2、Conv3_3、Conv4_4这四个卷积层的参数来计算。内容损失根据Conv3_3层的参数来计算。

原始风格迁移具体过程如下图，选定风格图片与即在一开始现随机生成一张噪点图片。之后对每一次更新过程，先计算生成图片的损失函数，后用梯度下降最小化损失函数，并更新生成函数中每一个像素点。
<img src="https://s1.ax1x.com/2020/10/05/0tw7FO.png" alt="0tw7FO.png" border="0" width=70%/>

这样每生成一张图片，就是一个“训练”过程。 在实际测试中，往往耗时漫长。根据图片大小及电脑性能，这一过程可能达到数个小时之久。

### 1.3 快速风格迁移[<sup>2</sup>](#refer-anchor-2)

传统风格迁移算法在生成一张图片是往往需要较长时间。主要是因为这种图像生成算法本质是一种“训练”的过程，计算量大且占用大量内存。那么一个很自然的想法就出现了：如果我们不把生成图片作为一个”训练“过程。而把生成图片作为一种计算的过程。在Johnson的论文中，他首次提出了快速风格迁移的概念。主要通过一个卷积神经网络来生成图片，并用VGG16来计算风格损失与内容损失。 之后通过梯度下降的方式更新图像生成网络的参数。具体如下图

<img src="https://s1.ax1x.com/2020/10/06/0U2oDO.png" alt="0U2oDO.png" border="0" width=70%/>


以上为论文中的图片，整个训练及生成的过程包含两个网络，**图像生成网络** 与**损失网络** 

<img src="https://s1.ax1x.com/2020/10/06/0UR5Js.png" alt="0UR5Js.png" border="0"  width=70%/>


* **图片生成网络**
图像变换网络总体也属于一个残差网络。一共是由3个卷积层、5个残差块、3个卷积层构成。这里没有用到池化等操作进行采用，在开始卷积层中（第二层、第三层）进行了下采样，在最后的3个卷积层中进行了上采样，将图片恢复到256*256的大小，同时将数值归一化到（1，255）来表示RGB颜色。以进入下一步VGG16网络计算

<img src="https://s1.ax1x.com/2020/10/07/0au9II.png" alt="0au9II.png" border="0" width=70% />

* **损失网络**

损失网络的主要功能是计算风格损失和内容损失。在每一次训练中，我们使用一张训练集中的照片作为内容图片。先将其通过图片生成网络生成一张新的图片，再将**内容图片**和**生成图片**经过VGG16的计算，在取出他们在Conv2-2层中激活值，计算内容损失。同理，使用**风格图片**与**生成图片**在Conv1-2 Conv2-2 Conv3-3 Conv4-4的激活值来计算风格损失。
<img src="https://s1.ax1x.com/2020/10/07/0aVT9U.png" alt="0aVT9U.png" border="0" width=70% />

* **训练过程**

在训练过程中，选定一张风格图片不变，选取训练集中的图片作为内容图片。训练集采用MSCOCO2014公开数据集进行训练，其包含月58万张图片。可以是算法充分学习到风格图片中的风格。
### 1.4 对比
**在损失函数的定义上，原始风格迁移与快速风格迁移相同。**

| **算法**         | 是否需要训练网络         | 预训练网络 | 图像生成方式                                   | **生成时间** |      |
| ---------------- | ------------------------ | ---------- | ---------------------------------------------- | ------------ | ---- |
| **原始风格迁移** | 否，                     | VGG19      | 通过梯度下降，最小化生成函数。<br>直接生成图像 | 数小时       |      |
| **快速风格迁移** | 是，需要训练图像生成网络 | VGG16      | 将内容图片输入网络，<br>计算后生成新图片       | 数秒         |      |




**因为快速风格迁移在训练时，风格图片是固定的。这意味着每一个模型仅对应一种风格。所以若要生成多种风格的图片，我们需要训练多个模型**
**为了

* **下载并读取VGG16预训练网络**
地址：'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

## 2.快速风格迁移复现

**使用Keras深度学习框架进行构建与训练**
### 2.1 生成网络与损失网络
#### 2.1.1 图像生成网络
图像生成网络的具体结构由李飞飞教授在论文中给出[<sup>3</sup>](#refer-anchor-3)
<img src="https://s1.ax1x.com/2020/10/07/0dkxiT.png" alt="0dkxiT.png" border="0" width=70%/>

其中 残差块具体结构如下

<img src="https://s1.ax1x.com/2020/10/07/0dEZBn.png" alt="0dEZBn.png" border="0" width=70%/>
左侧为此网络中使用的残差块，右侧为正常

* **Keras自定义层**
在快速风格迁移任务中，为了方便编程，需要在网络中加入一些自定义层（如归一化处理）。代码以class的方式定义，需要严格按照Keras手册中的方式


```python
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Deconvolution2D,  Conv2D,UpSampling2D,Cropping2D,Conv2DTranspose
from keras.layers import add
from keras.engine.input_layer import Input
```


```python
class Input_Norm(Layer):# 归一化使图片矩阵的值为（0，1）之间
    def __init__(self, **kwargs):
        super(InputNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_output_shape(self,input_shape):
        return input_shape

    def call(self, x, mask=None):
        return x/255.
class Denorm(Layer):# 逆归一化 将卷积层出的分布在（-1，1）之间的结果 变为（0，255）之间

    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass
    def compute_output_shape(self,input_shape):
        return input_shape
    def call(self, x, mask=None):
        return (x + 1) * 127.5

def res_block(x):
    y = x
    x = Conv2D(128,kernel_size = (3,3),strides = (1,1),padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128,kernel_size = (3,3),strides = (1,1),padding = 'same')(x)
    x = BatchNormalization()(x)
    res = add([x, y])
    return res
def img_transform_net():
    input = Input(shape=(256,256,3))
    input = Input_Norm()(input)
    #第一层卷积层
    x =Conv2D(32, kernel_size = (9,9), strides = (1,1), padding = 'same')(input)
    x =BatchNormalization()(x)
    x =Activation('relu')(x)
    #第二层卷积层
    x =Conv2D(64, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
    x =BatchNormalization()(x)
    x =Activation('relu')(x)
    #第三层卷积层
    x =Conv2D(128, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
    x =BatchNormalization()(x)
    x =Activation('relu')(x)
    #残差网络
    x = res_block(x)
    x = res_block(x)
    x = res_block(x)
    x = res_block(x)
    x = res_block(x)
    # 上采样
    x =Conv2DTranspose(64, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
    x =BatchNormalization()(x)
    x =Activation('relu')(x)
    #上采样
    x =Conv2DTranspose(32, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    #上采样
    x =Conv2DTranspose(3, kernel_size = (9,9), strides = (1,1), padding = 'same')(x)
    x =BatchNormalization()(x)
    output = layers.Activation('tanh')(x)  
    # 逆归一化
    output= Denormalize()(output)
    #定义模型
    model = Model(inputs = input,outputs = output)  
    return model
```

#### 2.1.2 损失网络
* **gram矩阵**


```python
def gram_matrix(x):#参考Keras example
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
```

* **风格损失与内容损失**

在这个过程中我们需要自定义损失函数。自定义loss层作为网络一层加进model，同时该loss的输出作为网络优化的目标函数


```python
from keras import backend as K
from keras.regularizers import Regularizer
from keras.objectives import mean_squared_error

class Style_loss(Regularizer):# 计算某一层的风格损失

    def __init__(self, style_feature_target, weight=1.0):
        self.style_feature_target = style_feature_target
        self.weight = weight
        self.uses_learning_phase = False
        super(StyleReconstructionRegularizer, self).__init__()

        self.style_gram = gram_matrix(style_feature_target)# 按照手册的方式定义

    def __call__(self, x):
        output = x.output[0]
        style_loss = self.weight *  K.sum(K.mean(K.square((self.style_gram-gram_matrix(output) )))) 

        return style_loss
class Content_loss(Regularizer):#计算某一行的内容损失
    def __init__(self, weight=1.0):
        self.weight = weight
        self.uses_learning_phase = False
        super(FeatureReconstructionRegularizer, self).__init__()

    def __call__(self, x):
        generated = x.output[0] 
        content = x.output[1] 

        loss = self.weight *  K.sum(K.mean(K.square(content-generated)))
        return loss
```


```python
from PIL import Image
import numpy as np
def Image_PreProcessing(path,img_width, img_height):
# 待处理图片存储路径	
    im = Image.open(path)
    
    imBackground = im.resize((img_width, img_height))
   
    re_img = np.asarray(imBackground)
    return re_img
```


```python


def compute_style_loss(vgg,style_image_path,vgg_layers,vgg_output_dict,weight):
    style_img = Image_PreProcessing(style_image_path, 256, 256)
    

    style_loss_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']# VGG16中来计算

    style_layer_outputs = []

    for layer in style_layers:
        style_layer_outputs.append(vgg_output_dict[layer])

    vgg_style_func = K.function([vgg.layers[-19].input], style_layer_outputs)

    style_features = vgg_style_func([style_img])

    # Style Reconstruction Loss
    for i, layer_name in enumerate(style_layers):
        layer = vgg_layers[layer_name]

        feature_var = K.variable(value=style_features[i][0])
        style_loss = StyleReconstructionRegularizer(
                            style_feature_target=feature_var,
                            weight=weight)(layer)

        layer.add_loss(style_loss)

def compute_content_loss(vgg_layers,vgg_output_dict,weight):

    content_layer = 'block4_conv2'
    content_layer_output = vgg_output_dict[content_layer]

    layer = vgg_layers[content_layer]
    content_regularizer = FeatureReconstructionRegularizer(weight)(layer)
    layer.add_loss(content_regularizer)
```


```python
from keras.engine.topology import Layer

class VGG_Norm(Layer):# 在图片进入损失网络前归一化
    def __init__(self, **kwargs):
        super(VGGNormalize, self).__init__(**kwargs)
    def build(self, input_shape):# 手册中要求 无实际意义
        pass
    def call(self, x):
        x = x[:, :, :, ::-1]       
        x = x-120
        return x
    
def dummy_loss(y_true, y_pred ):# 综合损失函数
    return K.variable(0.0)


```


```python
from keras.applications.vgg16 import VGG16
from keras.layers.merge import concatenate
transfer_model = img_transform_net()

tensor = concatenate([transformer.output,transformer.input],axis = 0)
tensor = VGGNormalize(name="vgg_normalize")(tensor)

vgg = VGG16(include_top = False,input_tensor = tensor2,weights = None)
vgg.load_weights("/Users/zcl271828/Downloads/fst-2/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",by_name = True)

vgg_output_dict = dict([(layer.name, layer.output) for layer in vgg.layers[-18:]])
vgg_layers = dict([(layer.name, layer) for layer in vgg.layers[-18:]])
```

* 将生成网络与损失网络结合，为最终训练做准备


```python
add_style_loss(vgg,'images/style/starry_night.jpg' , vgg_layers, vgg_output_dict,3)   
add_content_loss(vgg_layers,vgg_output_dict,2)
for layer in vgg.layers[-19:]:
    layer.trainable = False#固定损失网络参数
```

### 2.2 训练数据集

#### 2.2.1 下载并读取数据集

COCO数据集是一个大型的、丰富的物体检测，分割和字幕数据集。这个数据集以scene understanding为目标，主要从复杂的日常场景中截取。
下载地址：http://images.cocodataset.org/zips/train2014.zip

以下为训练集中的一张图片


```python
#
from matplotlib.pyplot import imshow
imshow(Image_PreProcessing("/Users/zcl271828/train2014/COCO_train2014_000000291797.jpg",256,256))

```




    <matplotlib.image.AxesImage at 0x7fcf38268750>




<img src="https://s1.ax1x.com/2020/10/07/0dcsQP.png" alt="0dcsQP.png" border="0" />


### 2.3 训练过程

#### 2.3.1 确定模型参数

由于计算资源有限，我们无法进行大规模试验去寻找最优超参数。因此，我们通过查阅资料的方式确定了最优超参数
* **batch size=1**
* **learing rate=1e-3**
* **训练次数=83000**

#### 2.3.2 进行训练


```python
from keras.preprocessing.image import ImageDataGenerator

optimizer = keras.optimizers.Adam()#。使用Adam优化器进行训练 超参数选为默认 可有效避免过拟合
vgg1.compile(optimizer,loss=dummy_loss)
gen = ImageDataGenerator()
train_data = gen.flow_from_directory('/fst-2/image/training_data',batch_size = 1,target_size = (256,256),class_mode = 'input')
# Keras针对图像训练的API 自动生成训练数据
history= vgg.fit_generator(train_data,steps_per_epoch=83000,epochs=1)
vgg.save_weights('starry_night_weights.h5')
vgg.save('starry_night.h5')# 储存模型



```

#### 2.3.3 训练过程
因个人电脑计算资源有限，我们在网络上找到了用于深度学习的GPU服务器。所以我们租用了价格为3元/小时的深度学习服务器。具体配置为GTX1080 TI 每个模型训练耗时大约在5-6小时，共训练7个模型。下图为控制台截图
<img src="https://s1.ax1x.com/2020/10/07/0dBGMF.png" alt="0dBGMF.png" border="0"  width=60% />

训练完成后，我们将模型文件下载到本地，在上传到我们自己的服务器上。为服务器读取模型并生成图片进行下一步准备

# 3. 云服务器API搭建

## 3. 云服务器API搭建

在部署模型过程中，考虑到训练模型较大，且需要有较高算力的计算设备，我们最终选择将模型搭建在云服务器，而并非直接部署在微信小程序。这保证了模型能在较短的时间内计算出来，并且拥有足够的内存空间供模型进行运算。

### 3.1 服务器类型比较

风格迁移模型在运算过程中，对设备内存和cpu有着较高的需求。使用tensorflow过程中会占用大量的内存空间，这就要求我们在选购服务器过程中，要充分考虑模型的需求。

目前市面上的服务器，在考虑费用与实际需求的基础上，可供我们选择的有**通用型服务器**与**突发型服务器**。

通用型云服务器ECS适用于高网络包收发场景，如视频弹幕、电信业务转发、企业级应用等应用场景。

突发型服务器适用于Web应用服务器、开发测试压测服务应用，不适用于长时间超过性能“基线”的需求场景或企业计算性能需求场景。

| 服务器 | 主机配置 | CPU性能  | 价格  |
|  ----  | ----  | ----  | ----  |
| 通用型服务器 | 2核4G | 100%使用 | 248元/月 |
| 突发型服务器 | 2核4G | 15%使用 | 54元/月 |


考虑到我们在使用服务器的过程中，仅会在处理图片时占用服务器，而且处理图片数量，在其他时间均处于空闲状态。因此我们最终决定选用突发型服务器，作为我们的云服务器进行风格迁移。

### 3.2 搭建flask框架

因为该项目代码均为python实现，所以在搭建web框架时我们选用python的flask框架。

flask首先接收到微信小程序发来的图片，并将图片存储到云服务器上；接着调用风格迁移模型，进行图片计算，并将计算好的图片储存；最后flask将生成的图片以返回值的形式，发送回微信小程序。



**以下代码均运行在我们自己的服务器上，接口：https://experimentforzcl.cn:8080**

#### 3.2.1 读取模型并生成图片



```python
from imageio import imwrite,imread
import numpy as np
from PIL import Image
import nets
import os
import string
import random
import json
import requests
from flask import Flask, request, redirect, url_for, render_template,send_file
import base64
# 裁减图片
def crop_image(img):
    aspect_ratio = img.shape[1]/img.shape[0]
    if aspect_ratio >1:
        w = img.shape[0]
        h = int(w // aspect_ratio)
        img =  K.eval(tf.image.crop_to_bounding_box(img, (w-h)//2,0,h,w))
    else:
        h = img.shape[1]
        w = int(h // aspect_ratio)
        img = K.eval(tf.image.crop_to_bounding_box(img, 0,(h-w)//2,h,w))
    return img

def main(style,input_file,out_name,original_color,blend_alpha,media_filter):
    img_width = img_height =  256
    #input_file="images/content/"+input_file_name+".jpg"
    out_put="images/generated/"+out_name+"_out.jpg"
    net = nets.image_transform_net(img_width,img_height)
    model = nets.loss_net(net.output,net.input,img_width,img_height,"",0,0)
    model.compile(Adam(),  dummy_loss) 
    model.load_weights("pretrained/"+style+'_weights.h5',by_name=False)
    y = net.predict(x)[0]
    y = crop_image(y)
    ox = crop_image(x[0], aspect_ratio)
    if blend_alpha > 0:
        y = blend(ox,y,blend_alpha)
    if original_color > 0:
        y = original_colors(ox,y,original_color )
    imwrite(out_put, y)
    return out_put


```

在查阅资料中，我们发现在在模型生成图片后，进行一些处理可以使图片有不同的效果。在这里我们选取了颜色保留与风格占比两种效果 对模型生成的图片进行处理[<sup>4</sup>](#refer-anchor-4)


```python
def original_colors(original, stylized,original_color):#内容图片颜色保留
    ratio=1. - original_color 

    hsv = color.rgb2hsv(original/255)
    hsv_s = color.rgb2hsv(stylized/255)

    hsv_s[:,:,2] = (ratio* hsv_s[:,:,2]) + (1-ratio)*hsv [:,:,2]
    img = color.hsv2rgb(hsv_s)    
    return img

def blend(original, stylized, alpha):#风格图片占比
    return alpha * original + (1 - alpha) * stylized
```

#### 3.2.2 提供API接口



```python
@app.route('/', methods=['POST'])
def index():
    # 接收对应参数
    return_dict= {'return_code': '404',"return_info":"xxx"}
    get_Data=request.form.to_dict()
    style=get_Data["style"]
    original_color=float(get_Data["original_color"])
    blend_alpha=float(get_Data["blend_alpha"])
    media_filter=3
    if request.method == 'POST':
        # 保存原始图片
        uploaded_file = request.files['file']
        print(uploaded_file)
        if uploaded_file.filename != '':
            if uploaded_file.filename[-3:] in ['jpg', 'png']:
                path_new,name=generate_filename()
                image_path = os.path.join("images/content", path_new)
                print(image_path)
                uploaded_file.save(image_path)
                print("ready")
                # 调用风格迁移函数
                with graph.as_default():
                    out_path=main(style,image_path,name,original_color,blend_alpha,media_filter)
                # 保存生成图片
                f = open(out_path, "rb")
                res = f.read()
                s = base64.b64encode(res)
                # 传回云服务器
                return s
    return json.dumps(return_dict)
```

### 3.3 配置Nginx与uwsgi

在配置服务器时，我们还使用了Nginx与uwsgi，通过flask+Nginx+uwsgi的服务器web部署，能够使程序在运行时更好的工作，进行任务的合理分配。

Nginx擅长处理静态请求，uwsgi擅长处理动态请求。当我们发起post请求后，首先被Nginx接收并分析，如果是动态请求，Nginx则通过socket把请求转向uwsgi去处理；如果是静态请求，Nginx则自行处理并将处理结果直接返回个客户端，这样基本上完成了一个完整的请求过程。

以下是Nginx配置代码：

``` 
server {
  listen 8080;
  server_name  www.experimentforzcl.cn;

  ssl on;
  ssl_certificate cert/4062906_www.experimentforzcl.cn.pem;
  ssl_certificate_key cert/4062906_www.experimentforzcl.cn.key;
  ssl_session_timeout 5m;
  ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
  ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:HIGH:!aNULL:!MD5:!RC4:!DHE;#°
  ssl_prefer_server_ciphers on;

  location / {
          include uwsgi_params;
          uwsgi_pass 127.0.0.1:6000;
  }
}

```

uwsgi可以应对多个用户同时请求的问题，将程序多线程处理，以下是配置代码：

```
[uwsgi]
socket = 127.0.0.1:6000
pythonpath = /root/experiment01/fst-2
wsgi-file = /root/experiment01/fst-2/app.py
callable = app
processes = 4
threads = 4
```



## 4. 微信小程序简介

作为一款深度学习为基础的照片处理产品，营造良好的用户体验是十分重要的。在微信日益普及的今天，微信小程序具有便捷性的特点，凭借其较低的开发难度以及较为普遍的使用，拥有着相当数量的用户。

因此本项目选择微信小程序，进行用户交互界面的搭建。通过微信小程序上传用户所选择的图片，利用服务器进行风格迁移后，再将处理好的图片返回微信小程序进行展示。

微信小程序页面主要分为以下几个部分：

<img src="https://s1.ax1x.com/2020/10/06/0NaeVf.png"/>

### 4.1 图片风格迁移

首先是图片风格迁移过程，为小程序主体页面，用户通过上传对应照片，进行相关风格的迁移。具体流程如下：

*进入风格迁移—选择对应风格—选择照片及参数—生成照片—完成风格迁移*

<table>
    <tr>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0Nd5hF.png"  width="80%" > </center></td>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0NdjAK.png"  width="80%"></center></td>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0Ndqn1.png"  width="80%"></center></td>
    </tr>
    <tr>
        <td><center>1.进入风格迁移</center></td>
        <td><center>2.选择对应风格</center> </td>
        <td><center>3.选择照片及参数</center> </td>
    </tr>
    <tr>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0NdHXR.png"  width="80%" > </center></td>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0NdT1J.png"  width="80%"></center></td>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0NwHPS.png"  width="80%"></center></td>
    </tr>
    <tr>
        <td><center>4.等待迁移完成</center></td>
        <td><center>5.迁移成功！</center> </td>
        <td><center>6.对应效果展示</center> </td>
    </tr>
</table>
微信小程序端风格迁移，利用微信API函数wx.uploadFile()，进行图片以及参数的上传；图片生成好后，再将生成的结果储存在微信云开发数据库中，用于历史记录的查询。

以下是部分代码展示：

```javascript
// 照片生成函数
generate: function() {
    var _this = this
    if(this.data.list[_this.data.current].choose==false){ 
      wx.showToast({
        title: '请选择图片',
        icon: 'none',
      })
    }else{
      var numvalue = _this.data.current
      wx.showLoading({
        title: '生成图片中...',
      })
      _this.setData({
        changeStyle: true,
      })
      wx.uploadFile({
        url: 'https://experimentforzcl.cn:8080',
        filePath: _this.data.list[_this.data.current].tempimg,
        name: 'file',
        formData:{'style': _this.data.modellist[_this.data.current],
            'original_color': Number(_this.data.original_color),
            'blend_alpha': Number(_this.data.blend_alpha)
          },
        success (res){
          if(res.statusCode==200){
            var temp1 = 'list[' + numvalue + '].proUrl'
            var temp2 = 'list[' + numvalue + '].generate'
            _this.setData({
              [temp1]: "data:image/png;base64," + res.data,
              [temp2]: true,
              changeStyle: false,
            })

            var number = Math.random();
            var timestamp = Date.parse(new Date());
            var date = new Date(timestamp);
            var time = date.getFullYear() + '-' + (date.getMonth() + 1 < 10 ? '0' + (date.getMonth() + 1) : date.getMonth() + 1) + '-' + (date.getDate() < 10 ? '0' + date.getDate() : date.getDate()) + "\t" + (date.getHours() < 10 ? '0' + date.getHours() : date.getHours()) + ":" + (date.getMinutes() < 10 ? '0' + date.getMinutes() : date.getMinutes())

            wx.getFileSystemManager().writeFile({
              filePath: wx.env.USER_DATA_PATH + '/pic' + number + '.png',
              data: _this.data.list[numvalue].proUrl.slice(22),
              encoding: 'base64',
            })
            wx.cloud.uploadFile({
              cloudPath: 'history/'+number+'.png',
              filePath: wx.env.USER_DATA_PATH + '/pic' + number + '.png',
              success: res => {
                db.collection('history').add({
                  data: {
                    time: time,
                    imgsrc: 'cloud://charlie-9mgmr.6368-charlie-9mgmr-1301103640/history/'+number+'.png',
                    style: numvalue,
                  }
                })
              },
            })
            wx.hideLoading(),
            Notify({
              type: 'success',
              message: '生成成功',
              duration: 1000,
            });
          }
        }
      })
    }
  },
```

此外，用户可以对生成的图片进行分享，或生成相关内容海报，提高小程序的知名度及影响力。

<table>
    <tr>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0Nc9mQ.png"  width="80%" > </center></td>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0N6z6S.png"  width="80%"></center></td>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0NcSOg.png"  width="80%"></center></td>
    </tr>
    <tr>
        <td><center>1.点击分享</center></td>
        <td><center>2.分享到微信</center> </td>
        <td><center>3.生成相关海报</center> </td>
    </tr>
</table>
### 4.2 图片风格介绍

用户可以查看每种风格的相关介绍，对每种生成的风格有一定了解。具体流程如下：

*点击风格周边—点击对应风格—查看风格介绍*

<table>
    <tr>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0Nd5hF.png"  width="80%" > </center></td>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0NsBt0.png"  width="80%"></center></td>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0NsJ1S.png"  width="80%"></center></td>
    </tr>
    <tr>
        <td><center>1.进入风格周边</center></td>
        <td><center>2.选择对应风格</center> </td>
        <td><center>3.查看风格介绍</center> </td>
    </tr>
</table>

在风格周边中，分别介绍了六种风格的具体色彩构成以及相关线条结构，帮助用户在生成图像前，可以对相关风格有更好的了解，增加用户体验。

### 4.3 个人中心

个人中心包括了**历史记录**、**意见反馈**以及**团队介绍**页面。

在历史记录中，可以查看过去成功生成的照片记录，包括了生成照片、生成照片风格及生成时间。

意见反馈包括了小程序客服页面，可以及时的提出相关意见反馈。

团队介绍里包括了团队的具体信息，以及团队联系方式。

<table>
    <tr>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0Nd5hF.png"  width="80%" > </center></td>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0Ng8Ej.png"  width="80%"></center></td>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0Ng1bQ.png"  width="80%"></center></td>
    </tr>
    <tr>
        <td><center>1.进入个人中心</center></td>
        <td><center>2.个人中心页面</center> </td>
        <td><center>3.查看历史记录</center> </td>
    </tr>
    <tr>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0NgQKS.png"  width="80%" > </center></td>
        <td ><center><img src="https://s1.ax1x.com/2020/10/06/0NglDg.png"  width="80%"></center></td>
        <td ><center></center></td>
    </tr>
    <tr>
        <td><center>4.进行意见反馈</center></td>
        <td><center>5.查看团队介绍</center> </td>
        <td><center></center> </td>
    </tr>
</table>



## 5. Reference

<div id="refer-anchor-1"></div>
- [1] [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

<div id="refer-anchor-2"></div>
- [2] [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155v1.pdf)

<div id="refer-anchor-3"></div>
- [3] [Perceptual Losses for Real-Time Style Transfer and Super-Resolution: Supplementary Material]( https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf)

<div id="refer-anchor-4"></div>
- [4] [Github](https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py)


```python

```

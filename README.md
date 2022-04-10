# ImageSearchLightningCLIP
Using distilled CLIP model to deploy the android device

# （一）应用介绍

# （二）使用说明

## 	快速开始

```
下载Android Studio->下载zip或git glone ‘AI_ALBUM’文件夹->用AS打开项目->连接手机编译即可
```

## 目录树

```
"AI_ALBUM/app/src/"下的目录（关键文件有说明）
└─main
    │  AndroidManifest.xml--应用清单（组织各个页面）
    │  
    ├─assets--资源文件夹
    │      .gitattributes
    │      image_encode.ptl--图像编码器模型
    │      tiny_text_encode.ptl--文本编码器模型
    │      vocab.txt--文本词典
    │      
    ├─java--程序关键部分
    │  └─org
    │      └─pytorch
    │          └─helloworld
    │                  Bean.java--瀑布流实现必要类--存储item内容
    │                  MyAdapter.java--瀑布流实现必要类--适配器类
    │                  Image2ImageActivity.java--图片展示界面1
    │                  ImageActivity.java--拍照界面（图片推理）
    │                  MainActivity.java--主界面
    │                  ShareData.java--存储全局共享变量
    │                  Text2ImageActivity.java--图片展示界面2
    │                  TextActivity.java--文本框输入界面（文本推理）
    │                  Tokenizer.java--文本预处理类（string->token）
    │                  
    └─res--界面资源文件
        ├─drawable
        │      bg_main.png--UI背景图片1
        │      bg_vision.png--UI背景图片2
        │      ic_launcher_background.xml--图标设计
        │      round_corner.xml--图片展示外观设计
        │      search_box.xml--文本搜索框外观设计
        │      start_window.png--启动页图片
        │      
        ├─drawable-v24
        │      ic_launcher_foreground.xml
        │      
        ├─layout--对应java文件的UI布局文件
        │      activity_image.xml
        │      activity_main.xml
        │      activity_text.xml
        │      image_show.xml--瀑布流主文件
        │      list_item.xml--瀑布流子文件
        │      
        ├─mipmap-hdpi
        │      ic_launcher.png
        │      ic_launcher_round.png
        │      
        ├─mipmap-mdpi
        │      ic_launcher.png
        │      ic_launcher_round.png
        │      
        ├─mipmap-xhdpi
        │      ic_launcher.png
        │      ic_launcher_round.png
        │      
        ├─mipmap-xxhdpi
        │      ic_launcher.png
        │      ic_launcher_round.png
        │      
        ├─mipmap-xxxhdpi
        │      ic_launcher.png
        │      ic_launcher_round.png
        │      
        ├─values--主体设计文件
        │      colors.xml
        │      strings.xml
        │      styles.xml
        │      
        └─xml
                file_paths.xml--拍照图片的临时保存路径
```

## App使用演示

![show](C:\Users\Chao\Desktop\NLP\ImageSearchLightningCLIP\show.gif)

# （三）版本更新日志

## 1.0--AI ALBUM--2022.04.01

### 	实现功能：

​				实现基本的功能与界面设计；

### 	存在的问题：	

​				导入图片速度过慢；

​				文本搜索框暂无检测异常机制，容易造成程序停止运行；

​				词典没有扩充，内容覆盖范围不够；

​				初始界面无加载页面或动画，体验较差

## 2.0--AI ALBUM--2022.04.04
### 	实现功能：

​				优化载入图库效率，加载时间缩短为原来1/3；

​				添加了启动页面；

​				添加了文本框的异常检测；

​				优化了图片的展示布局；

### 	存在的问题：	

​				词典没有扩充，内容覆盖范围不够；

​				图片加载阻塞UI界面线程，影响体验；


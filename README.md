# ImageSearchLightningCLIP
Using distilled CLIP model to deploy the android device

## 1.0--版本-实现基本功能与界面
### 存在的问题：	
    ①初始界面的优化：异步加载+多线程导入图片+初始界面设计（加载条，提示语，演示动画）
    ②对搜索框进行异常的检测（防止出错就卡死）+完善字典的查询（bpe与特殊字符的过滤）

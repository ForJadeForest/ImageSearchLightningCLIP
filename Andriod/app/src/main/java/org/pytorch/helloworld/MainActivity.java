package org.pytorch.helloworld;

//安卓的库
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

//pytorch的库
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.MemoryFormat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    //继承onCreate函数
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    //定义bitmap和module两个变量
    Bitmap bitmap = null;
    Module module = null;
    Module module_image = null;
    Module module_text = null;
    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg 将jpg图片转换成bitmap
      /* 1-读取图片 */
      //Image image=getAssets().open("image.jpg");
      bitmap = BitmapFactory.decodeStream(getAssets().open("test2.jpg"));

      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      /* 2-读取模型 */
      module = LiteModuleLoader.load(assetFilePath(this, "model.pt"));
      //读取image模型
      module_image = LiteModuleLoader.load(assetFilePath(this, "our_image_clip_cpu.ptl"));
      //读取text模型
      //module_text = LiteModuleLoader.load(assetFilePath(this, "our_text_clip.ptl"));
      //module_image = LiteModuleLoader.load(assetFilePath(this, "clip.ptl"));
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }

    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);

    // preparing input tensor
    /* 3-将bitmap图片转换成tensor并进行预处理（需与原模型输入保持一直，可以保证准确率） */
    final int INPUT_TENSOR_WIDTH = 224;
    final int INPUT_TENSOR_HEIGHT = 224;
    bitmap=resizeImage(bitmap,INPUT_TENSOR_WIDTH,INPUT_TENSOR_HEIGHT);
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

    //final Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(bitmap,
     //       TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
    // running the model
    /* 4-将图片输入模型 */
    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();//forward(IValue.from(inputTensor)).toTensor();
    final Tensor outputImageFeature = module_image.forward(IValue.from(inputTensor)).toTensor();
    // getting tensor content as java array of floats
    final float[] scores = outputTensor.getDataAsFloatArray();
    final float[] ImageFeatures = outputImageFeature.getDataAsFloatArray();
    // searching for the index with maximum score
    /* 5-比较输出最大可能的分类 */
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    int[] max_id = new int[5];
    String[] className = new String[5];
    for (int i = 0; i < 5; i++)
    {
      maxScoreIdx=maxindex(scores);
      max_id[i]=maxScoreIdx;
      className[i]=ImageNetClasses.IMAGENET_CLASSES[max_id[i]];
      scores[maxScoreIdx]=0;
    }


    // showing className on UI
    TextView textView = findViewById(R.id.text);
    textView.setText("                                 前五分类预测\n"+className[0]+"\n"+className[1]+"\n"+className[2]+"\n"+className[3]+"\n"+className[4]+"\n");
    //textView.setText("习近平");
  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  public static int maxindex(float[] arr) {

    int max = 0;
    for (int i = 1; i < arr.length; i++) {

      if (arr[max] < arr[i]) {
        max = i;
      }

    }
    return max;
  }
  public Bitmap resizeImage(Bitmap bitmap, int w, int h) {
    Bitmap BitmapOrg = bitmap;
    int width = BitmapOrg.getWidth();
    int height = BitmapOrg.getHeight();
    int newWidth = w;
    int newHeight = h;

    float scaleWidth = ((float) newWidth) / width;
    float scaleHeight = ((float) newHeight) / height;

    Matrix matrix = new Matrix();
    matrix.postScale(scaleWidth, scaleHeight);
    // if you want to rotate the Bitmap
    // matrix.postRotate(45);
    Bitmap resizedBitmap = Bitmap.createBitmap(BitmapOrg, 0, 0, width,height, matrix, true);
    return resizedBitmap;
  }

  public Tensor ConSimilarity(float[][] ImageFeatures,float[][] ImageFeatures)
  {
     for(int i=0;i<10;i++)
     {

     }
    ImageFeatures*ImageFeatures;
  }

}

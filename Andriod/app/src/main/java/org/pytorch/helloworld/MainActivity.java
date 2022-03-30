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
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import org.pytorch.helloworld.Tokenizer;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

  public static float[] TORCHVISION_NORM_MEAN_RGB = new float[] {0.481f, 0.458f, 0.408f};
  public static float[] TORCHVISION_NORM_STD_RGB = new float[] {0.269f, 0.261f, 0.276f};



  public MainActivity() throws IOException {
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    //继承onCreate函数
    super.onCreate(savedInstanceState);
    setContentView(R.layout.main_activaty_ablayerout);

    //定义bitmap和module两个变量
    List<Bitmap> ImageSet = new ArrayList<Bitmap>();
    Bitmap bitmap = null;
    Module module_image = null;
    Module module_text = null;
    Tokenizer Simple_tokenize = null;
    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg 将jpg图片转换成bitmap
      /* 1-读取图片 */
      //Image image=getAssets().open("image.jpg");
      bitmap = BitmapFactory.decodeStream(getAssets().open("test2.jpg"));
      ImageSet.add(BitmapFactory.decodeStream(getAssets().open("test1.jpg")));
      ImageSet.add(BitmapFactory.decodeStream(getAssets().open("test2.jpg")));
      ImageSet.add(BitmapFactory.decodeStream(getAssets().open("test3.jpg")));
      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      /* 2-读取模型 */
      //module = LiteModuleLoader.load(assetFilePath(this, "model.pt"));
      Simple_tokenize = new Tokenizer(assetFilePath(this, "vocab.txt"));
      //读取image-text模型
      module_image = LiteModuleLoader.load(assetFilePath(this, "image_encode.ptl"));
      module_text = LiteModuleLoader.load(assetFilePath(this, "text_encode.ptl"));
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }

    // showing image on UI
    //ImageView imageView = findViewById(R.id.imageView2);
    //imageView.setImageBitmap(bitmap);
    TextView textView4 = findViewById(R.id.textView4);
    textView4.setText("lebron james");
    // preparing input tensor
    /* 3.1-将String文本转换进行预处理*/

    Tensor text_token = tokenize("lebron james", 77, Simple_tokenize);

    /* 3.2-将bitmap图片转换成tensor并进行预处理（需与原模型输入保持一直，可以保证准确率） */
    final int INPUT_TENSOR_WIDTH = 224;
    final int INPUT_TENSOR_HEIGHT = 224;
    //List<Tensor> TensorSet = new ArrayList<Tensor>();
    Tensor t_tensor = null;
    float[][] ImageFeatureSet = new float[ImageSet.size()][];
    for (int i = 0; i < ImageSet.size(); i++) {
      ImageSet.set(i, resizeImage(ImageSet.get(i), INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT));
      t_tensor = TensorImageUtils.bitmapToFloat32Tensor(ImageSet.get(i), TORCHVISION_NORM_MEAN_RGB, TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
      //TensorSet.add(TensorImageUtils.bitmapToFloat32Tensor(ImageSet.get(i),TORCHVISION_NORM_MEAN_RGB, TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST));

      t_tensor = module_image.forward(IValue.from(t_tensor)).toTensor();
      ImageFeatureSet[i] = t_tensor.getDataAsFloatArray();
    }

    bitmap = resizeImage(bitmap, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT);
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
            TORCHVISION_NORM_MEAN_RGB, TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

    //final Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(bitmap,
    //       TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
    // running the model
    /* 4-将图片输入模型 */
    //final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();//forward(IValue.from(inputTensor)).toTensor();
    final Tensor outputImageFeature = module_image.forward(IValue.from(inputTensor)).toTensor();
    final Tensor outputTextFeature = module_text.forward(IValue.from(text_token)).toTensor();
    // getting tensor content as java array of floats
    //final float[] scores = outputTensor.getDataAsFloatArray();
    final float[] ImageFeature = outputImageFeature.getDataAsFloatArray();
    final float[] TextFeature = outputTextFeature.getDataAsFloatArray();
    // searching for the index with maximum score
    /* 5-比较输出最大可能的分类 */
    float[] Similarity = ConSimilarity(ImageFeatureSet, TextFeature);
    TextView textView = findViewById(R.id.textView);
    TextView textView2 = findViewById(R.id.textView2);
    TextView textView3 = findViewById(R.id.textView3);

    ImageView imageView3 = findViewById(R.id.imageView3);
    ImageView imageView4 = findViewById(R.id.imageView4);
    ImageView imageView5 = findViewById(R.id.imageView5);
    int[] Index = Arraysort(Similarity);
    TextView textView1 = findViewById(R.id.text);
    textView.setText("Top[" + 0 + "]:" + String.format("%.2f", Similarity[0]));
    textView2.setText("Top[" + 1 + "]:" + String.format("%.2f", Similarity[1]));
    textView3.setText("Top[" + 2 + "]:" + String.format("%.2f", Similarity[2]));

    imageView3.setImageBitmap(ImageSet.get(Index[0]));
    imageView4.setImageBitmap(ImageSet.get(Index[1]));
    imageView5.setImageBitmap(ImageSet.get(Index[2]));
    /*
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

     */
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

  public float[] ConSimilarity(float[][] ImageFeatures,float[] TextFeatures)
  {
    float [] Similarity = new float[ImageFeatures.length];
    for(int i=0;i<ImageFeatures.length;i++)
    {
      for(int j=0;j<ImageFeatures[i].length;j++)
      {
        Similarity[i]+=ImageFeatures[i][j]*TextFeatures[j];
      }
    }
    float sum=0;
    for(int i=0;i<Similarity.length;i++)
    {
      Similarity[i]= (float) Math.exp(100*Similarity[i]);
      sum+=Similarity[i];
    }
    for(int i=0;i<Similarity.length;i++)
    {
      Similarity[i]/=sum;
    }
    return Similarity;
  }

  public static int[] Arraysort(float[]arr)
  {
    //double[] arr = {5.5,2,66,3,7,5};
    float temp;
    int index;
    int k=arr.length;
    int[]Index= new int[k];
    for(int i=0;i<k;i++)
    {
      Index[i]=i;
    }

    for(int i=0;i<arr.length;i++)
    {
      for(int j=0;j<arr.length-i-1;j++)
      {
        if(arr[j]<arr[j+1])
        {
          temp = arr[j];
          arr[j] = arr[j+1];
          arr[j+1] = temp;

          index=Index[j];
          Index[j] = Index[j+1];
          Index[j+1] = index;
        }
      }
    }
    return Index;
  }

  public Tensor tokenize(String text,int context_length,Tokenizer Simple_tokenize)
  {

    int startToken=Simple_tokenize.encoder.get("<|startoftext|>");
    int endToken=Simple_tokenize.encoder.get("<|endoftext|>");

    //int startToken=Tokenizer.encode();
    //int endToken=49407;

    //int [] textToken=Simple_tokenize.encode(text);
    int [] textToken=Tokenizer.encode(text);
    int [] allToken=new int[context_length];
    for(int i=0;i<context_length;i++)
    {
      if(i==0)
      {
        allToken[i]=startToken;
      }
      else if(i<textToken.length+1 && i>0)
      {
        allToken[i]=textToken[i-1];
      }
      else if(i==textToken.length+1) {
        allToken[i] = endToken;
      }
    }
    Tensor inputTensor = Tensor.fromBlob(allToken, new long[]{1, context_length});
    return inputTensor;
  }
}

package org.pytorch.helloworld;

//安卓的库
import static android.os.Environment.DIRECTORY_DCIM;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
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
import androidx.core.app.ActivityCompat;

import com.xiasuhuei321.loadingdialog.view.LoadingDialog;

public class MainActivity extends AppCompatActivity {

  public static float[] TORCHVISION_NORM_MEAN_RGB = new float[] {0.481f, 0.458f, 0.408f};
  public static float[] TORCHVISION_NORM_STD_RGB = new float[] {0.269f, 0.261f, 0.276f};
  public static int INPUT_TENSOR_WIDTH = 224;
  public static int INPUT_TENSOR_HEIGHT = 224;

  private List<Bitmap> ImageSet = new ArrayList<Bitmap>();
  private List<File> OriginImageSet=new ArrayList<>();

  public MainActivity() throws IOException {
  }

  private class MyThread_load_image extends Thread
  {
    private int start = 0;
    private int end = 100;

    public MyThread_load_image(int start,int end)
    {
      this.start=start;
      this.end=end;
    }

    @Override
    public void run() {
      for (int i = start; i < end; i++) {
        try {
          Bitmap bitmap = BitmapFactory.decodeFile(OriginImageSet.get(i).getPath(), getBitmapOption(4));
          ImageSet.add(bitmap);
        } catch (OutOfMemoryError e) {
        }
      }
    }
  }




  @Override
  protected void onCreate(Bundle savedInstanceState) {
    //继承onCreate函数
    super.onCreate(savedInstanceState);
    setTheme(R.style.AppTheme);//恢复原有的样式
    setContentView(R.layout.activity_main);

    getSupportActionBar().hide();
    //去掉最上面时间、电量等
    this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

    findViewById(R.id.Textbutton).setOnClickListener(v -> startActivity(new Intent(MainActivity.this, TextActivity.class)));
    findViewById(R.id.Imagebutton).setOnClickListener(v -> startActivity(new Intent(MainActivity.this, ImageActivity.class)));





    if(Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED) )
    {
      // 检查读取和写入动态权限
      if(ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED)
      { }
      else
      {
        // 如果没有，获取读取和写入动态权限
        ActivityCompat.requestPermissions(MainActivity.this,new String[]{ Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE},100 );
      }
    }

    String ImagePath=Environment.getExternalStoragePublicDirectory(DIRECTORY_DCIM).toString()+"/Camera/";
    File Imagefile = new File(ImagePath);
    File [] fileSet =Imagefile.listFiles();


    for(File file:fileSet ){
      if(file.getName().endsWith(".jpg") || file.getName().endsWith(".png"))
      {
        OriginImageSet.add(file);
      }
      Log.d(""," 文件名："+file.getName()+"文件路径 ："+file.getAbsolutePath());
    }
    //imageView.setImageURI(Uri.fromFile(OriginImageSet.get(0)));


    MyThread_load_image[] myLoadImageThread=Generate_ThreadGroup(8,OriginImageSet.size());
    LoadingDialog ld = new LoadingDialog(this);
    ld.setLoadingText("加载中")
            .setSuccessText("成功加载图库")//显示加载成功时的文字
            //.setFailedText("加载失败")
            .show();
//在你代码中合适的位置调用反馈
//ld.loadFailed();

    for(MyThread_load_image i:myLoadImageThread)
    {
      i.start();
    }
    try {
      for(MyThread_load_image i:myLoadImageThread)
      {
        i.join();
      }
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
/*
    for(int i=0;i<OriginImageSet.size();i++)
    {
      try {
        Bitmap bitmap = BitmapFactory.decodeFile(OriginImageSet.get(i).getPath(), getBitmapOption(4));
        ImageSet.add(bitmap);
      }
      catch (OutOfMemoryError e)
      { }
    }

 */

    final ShareData sharedata = (ShareData)getApplication();
    List<Bitmap> ShareImageSet=new ArrayList<>(ImageSet);
    sharedata.setImageSet(ShareImageSet);

    //定义bitmap和module两个变量


    Module module_image = null;
    Module module_text = null;

    try {
      module_image = LiteModuleLoader.load(assetFilePath(this, "image_encode.ptl"));
      module_text = LiteModuleLoader.load(assetFilePath(this, "tiny_text_encode.ptl"));
      sharedata.setModule_image(module_image);
      sharedata.setModule_text(module_text);
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }

    // preparing input tensor
    /* 3.2-将bitmap图片转换成tensor并进行预处理（需与原模型输入保持一直，可以保证准确率） */


    Tensor t_tensor = null;
    float[][] ImageFeatureSet = new float[ImageSet.size()][];
    for (int i = 0; i < ImageSet.size(); i++)
    {
      ImageSet.set(i, resizeImage(ImageSet.get(i), INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT));
      t_tensor = TensorImageUtils.bitmapToFloat32Tensor(ImageSet.get(i), TORCHVISION_NORM_MEAN_RGB, TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
      t_tensor = module_image.forward(IValue.from(t_tensor)).toTensor();
      ImageFeatureSet[i] = t_tensor.getDataAsFloatArray();
    }
    ld.loadSuccess();
    sharedata.setImageSetFeature(ImageFeatureSet);
  }



  private BitmapFactory.Options getBitmapOption(int inSampleSize)
  {
    System.gc();
    BitmapFactory.Options options = new BitmapFactory.Options();
    options.inJustDecodeBounds = true;

    options.inPurgeable = true;
    options.inSampleSize = inSampleSize;
    options.inJustDecodeBounds = false;
    return options;
  }

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
  public MyThread_load_image[] Generate_ThreadGroup(int num_thread, int imageSize)
  {
    MyThread_load_image[] myThread=new MyThread_load_image[num_thread];
    int basesize=imageSize/num_thread;
    int lastsize=imageSize%num_thread;
    for(int i=0;i<myThread.length;i++)
    {
      if(i!=myThread.length-1)
      {
        myThread[i]=new MyThread_load_image(i*basesize,(i+1)*basesize);
      }
      else
      {
        myThread[i]=new MyThread_load_image(i*basesize,(i+1)*basesize+lastsize);
      }

    }
    return myThread;
  }

  public static float[] ConSimilarity(float[][] ImageFeatures,float[] TextFeatures)
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
      Similarity[i]= (float) Math.exp(10*Similarity[i]);
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
  public static Bitmap resizeImage(Bitmap bitmap, int w, int h) {
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

}

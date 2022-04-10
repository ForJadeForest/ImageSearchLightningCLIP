package org.pytorch.helloworld;

import static org.pytorch.helloworld.MainActivity.*;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;


import android.widget.Toast;
import org.pytorch.IValue;
import org.pytorch.MemoryFormat;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;


public class ImageActivity extends AppCompatActivity {

    private Bitmap bitmap=null;
    private static final int REQUEST_IMAGE = 1;
    private static final String[] authCameraArr = {Manifest.permission.WRITE_EXTERNAL_STORAGE};
    private static final int authCameraRequestCode = 5;
    String currentPhotoPath;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image);

        getSupportActionBar().hide();
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);


        if(Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED) )
        {
            // 检查读取和写入动态权限
            if(ActivityCompat.checkSelfPermission(ImageActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED)
            { }
            else
            {
                // 如果没有，获取读取和写入动态权限
                ActivityCompat.requestPermissions(ImageActivity.this,new String[]{ Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE},100 );
            }
        }

        startCamera();

        Button button_image = (Button) findViewById(R.id.find_similar);
        final ShareData sharedata = (ShareData)getApplication();

        button_image.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Intent intent = new Intent();

                bitmap=resizeImage(bitmap, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT);
                Tensor Imagetensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, TORCHVISION_NORM_MEAN_RGB, TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
                float[] Imagefeature=sharedata.getModule_image().forward(IValue.from(Imagetensor)).toTensor().getDataAsFloatArray();
                float[] ImageSimilarity=ConSimilarity(sharedata.getImageSetFeature(),Imagefeature);
                int[] ImageIndex=Arraysort(ImageSimilarity);

                sharedata.setSimilarity(ImageSimilarity);
                intent.putExtra("ImageIndex",ImageIndex);
                intent.setClass(ImageActivity.this,Image2ImageActivity.class);
                startActivity(intent);
            }
        });
    }




    //判断当前是否具备所需权限
    private boolean hasCameraPhoneAuth(){
        PackageManager pm = this.getPackageManager();
        for(String auth: authCameraArr){
            if(pm.checkPermission(auth, this.getPackageName()) != PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return true;
    }

    //调用系统相机
    public void startCamera(){
        //Android 6.0以上需要运行时权限
        if(Build.VERSION.SDK_INT >= 23){
            if(!hasCameraPhoneAuth()){
                this.requestPermissions(authCameraArr, authCameraRequestCode);
                return;
            }
        }

        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        //Android7.0文件保存方式改变了
        File photoFile = null;
        try {
            photoFile = createImageFile();
        } catch (IOException ex) {
        }
        // Continue only if the File was successfully created
        if (photoFile != null) {
            Uri photoURI = FileProvider.getUriForFile(this,
                    "com.example.android.fileprovider",
                    photoFile);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
            startActivityForResult(intent, REQUEST_IMAGE);
        }
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }

    private void galleryAddPic() {
        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        File f = new File(currentPhotoPath);
        Uri contentUri = Uri.fromFile(f);
        mediaScanIntent.setData(contentUri);
        this.sendBroadcast(mediaScanIntent);
    }

    private BitmapFactory.Options getBitmapOption(int inSampleSize)
    {
        System.gc();
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPurgeable = true;
        options.inSampleSize = inSampleSize;
        return options;
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode == RESULT_OK){
            if(requestCode == REQUEST_IMAGE){
                try {
                    galleryAddPic();
                    bitmap = BitmapFactory.decodeFile(currentPhotoPath,getBitmapOption(2));
                    ImageView imageView=findViewById(R.id.pic);
                    imageView.setImageBitmap(bitmap);
                }catch(Exception e){
                    e.printStackTrace();
                }
            }
        }
    }

    //运行时权限的回调
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode == authCameraRequestCode){
            for(int ret: grantResults){
                if(ret == PackageManager.PERMISSION_GRANTED){
                    continue;
                }else{
                    Toast.makeText(this, "缺少写文件的权限!", Toast.LENGTH_SHORT).show();
                    return;
                }
            }
            startCamera();
        }
    }

}
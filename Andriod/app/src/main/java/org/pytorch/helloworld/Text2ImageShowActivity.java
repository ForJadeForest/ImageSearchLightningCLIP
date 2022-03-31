package org.pytorch.helloworld;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

public class Text2ImageShowActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text2_image_show);

        getSupportActionBar().hide();
        //去掉最上面时间、电量等
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        final ShareData sharedata = (ShareData)getApplication();
        Intent intent =getIntent();
        //getXxxExtra方法获取Intent传递过来的数据
        String text=intent.getStringExtra("text");
        int[] Top=intent.getIntArrayExtra("Index");
        TextView textView = findViewById(R.id.textView);
        ImageView imageView1 = findViewById(R.id.imageView1);
        ImageView imageView2 = findViewById(R.id.imageView2);
        ImageView imageView3 = findViewById(R.id.imageView3);

        textView.setText(text);
        imageView1.setImageBitmap(sharedata.getImageSet().get(Top[0]));
        imageView2.setImageBitmap(sharedata.getImageSet().get(Top[1]));
        imageView3.setImageBitmap(sharedata.getImageSet().get(Top[2]));
    }
}
package org.pytorch.helloworld;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.recyclerview.widget.StaggeredGridLayoutManager;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;


public class Image2ImageActivity extends AppCompatActivity {

    private final List<Bean> data = new ArrayList<>();
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
        int[] Top=intent.getIntArrayExtra("ImageIndex");
        float[] ImageSimilarity=sharedata.getSimilarity();
        float threshold=0;
        int Showsize=0;

        for(int i=0;i<ImageSimilarity.length;i++)
        {
            if(ImageSimilarity[i]<0.005)
            {
                Showsize=i;
                break;
            }
        }

        if(Showsize<10)
        {
            Showsize=10;
        }

        //初始化页面数据
        for (int i = 0;i<Showsize;i++){
            Bean bean = new Bean();
            //bean.setHeight((int) (Math.random() * 300 + 200));
            bean.setImage(sharedata.getImageSet().get(Top[i]));
            data.add(bean);
        }

        /***
         * recyclerview 需要 manager ,确定布局方式 可以为GridLayoutManager，LinearLayoutManager 等布局方式的Manager
         * StaggeredGridLayoutManager 是一个瀑布流方式的manager
         */

        StaggeredGridLayoutManager manager = new StaggeredGridLayoutManager(1,StaggeredGridLayoutManager.VERTICAL);
        RecyclerView recyclerView = findViewById(R.id.list);
        MyAdapter myAdapter = new MyAdapter(data,this);
        //添加 adapter
        recyclerView.setAdapter(myAdapter);
        //添加 manager
        //recyclerView.setLayoutManager(new LinearLayoutManager(Image2ImageActivity.this,LinearLayoutManager.VERTICAL,false));
        recyclerView.setLayoutManager(manager);

    }
}


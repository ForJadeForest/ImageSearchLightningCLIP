package org.pytorch.helloworld;

import static org.pytorch.helloworld.MainActivity.*;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.widget.EditText;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Tensor;

import java.io.IOException;

public class TextActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text);

        getSupportActionBar().hide();
        //去掉最上面时间、电量等
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        try {
            final Tokenizer Simple_tokenize = new Tokenizer(assetFilePath(this, "vocab.txt"));
            final EditText editText = (EditText) findViewById(R.id.textinput);
            final ShareData sharedata = (ShareData)getApplication();

            editText.setOnEditorActionListener(new TextView.OnEditorActionListener()
            {
                @Override
                public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                    if (actionId == EditorInfo.IME_ACTION_SEARCH) {
                        //搜索（发送消息）操作（此处省略，根据读者自己需要，直接调用自己想实现的方法即可）
                        String text=editText.getText().toString();

                        Tensor textTensor=tokenize(text,77,Simple_tokenize);
                        float[] Textfeature =sharedata.getModule_text().forward(IValue.from(textTensor)).toTensor().getDataAsFloatArray();
                        float[] TextSimilarity=ConSimilarity(sharedata.getImageSetFeature(),Textfeature);
                        int[] Index=Arraysort(TextSimilarity);
                        
                        Intent intent = new Intent();
                        intent.putExtra("text", text);
                        intent.putExtra("Index", Index);
                        intent.setClass(TextActivity.this, Text2ImageShowActivity.class);
                        startActivity(intent);

                    }
                    return false;
                }
            });
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }


    public Tensor tokenize(String text, int context_length, Tokenizer Simple_tokenize)
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
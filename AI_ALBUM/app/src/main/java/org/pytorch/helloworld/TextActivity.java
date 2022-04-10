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
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Tensor;

import java.io.IOException;

public class TextActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text);

        getSupportActionBar().hide();
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

                        try
                        {
                            int[] allToken=tokenize(text,77,Simple_tokenize);
                            Tensor textTensor = Tensor.fromBlob(allToken, new long[]{1, 77});
                            float[] Textfeature =sharedata.getModule_text().forward(IValue.from(textTensor)).toTensor().getDataAsFloatArray();
                            float[] TextSimilarity=ConSimilarity(sharedata.getImageSetFeature(),Textfeature);
                            int[] Index=Arraysort(TextSimilarity);

                            sharedata.setSimilarity(TextSimilarity);
                            Intent intent = new Intent();
                            intent.putExtra("text", text);
                            intent.putExtra("Index", Index);
                            intent.setClass(TextActivity.this, Text2ImageActivity.class);
                            startActivity(intent);
                        }
                        catch(NullPointerException e)
                        {
                            Toast.makeText(TextActivity.this, "存在非法字符或单词，请重新输入!", Toast.LENGTH_SHORT).show();
                        }

                    }
                    return false;
                }
            });
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }


    public int[] tokenize(String text, int context_length, Tokenizer Simple_tokenize)
    {

        int startToken=Simple_tokenize.encoder.get("<|startoftext|>");
        int endToken=Simple_tokenize.encoder.get("<|endoftext|>");


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
        //Tensor inputTensor = Tensor.fromBlob(allToken, new long[]{1, context_length});
        return allToken;
    }
}
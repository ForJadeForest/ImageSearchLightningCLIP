package org.pytorch.helloworld;

import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class MyAdapter extends RecyclerView.Adapter<MyAdapter.MyViewHolder> {
    private List<Bean> data; //数据集合
    private Context context;//上下文对象

    public MyAdapter(List<Bean> data, Context context) {
        this.data = data;
        this.context = context;
    }

    @NonNull
    @Override
    public MyViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        //获取 layout list_item.xml
        View view = View.inflate(context, R.layout.list_item, null);
        return new MyViewHolder(view); //返回 holder ,用于管理当前瀑布流元素
    }

    @Override
    public void onBindViewHolder(@NonNull MyViewHolder holder, int position) {
        holder.im.setImageBitmap(data.get(position).getImage());
    }

    /***
     * 返回数据总条数
     * @return
     */
    @Override
    public int getItemCount() {
        return data == null ? 0 : data.size();
    }

    //管理瀑布流元素类
    public class MyViewHolder extends RecyclerView.ViewHolder {
        private ImageView im;
        public MyViewHolder(@NonNull View itemView) {
            super(itemView);
            //获取文本节点
            im= itemView.findViewById(R.id.im);
        }
    }
}

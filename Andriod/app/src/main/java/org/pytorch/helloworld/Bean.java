package org.pytorch.helloworld;

import android.graphics.Bitmap;

public class Bean {
    private String name; //名称
    private int height; //高度
    private Bitmap picture;



    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }


    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Bitmap getImage() {
        return picture;
    }

    public void setImage(Bitmap pic) {
        this.picture = pic;
    }
}



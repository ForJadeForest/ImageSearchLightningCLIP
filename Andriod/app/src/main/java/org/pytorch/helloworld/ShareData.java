package org.pytorch.helloworld;

import android.os.Bundle;

import android.app.Application;
import android.graphics.Bitmap;
import android.os.Bundle;

import org.pytorch.Module;

import java.util.ArrayList;
import java.util.List;

public class ShareData extends Application {

    private List<Bitmap> ImageSet;

    private float[][] ImageSetFeature;
    private Module module_image;
    private Module module_text;
    private float[] Similarity;

    public List<Bitmap> getImageSet()
    {
        return this.ImageSet;
    }
    public void setImageSet(List<Bitmap> set)
    {
        this.ImageSet= set;
    }

    public float[][] getImageSetFeature()
    {
        return this.ImageSetFeature;
    }
    public void setImageSetFeature(float[][] feature)
    {
        this.ImageSetFeature= feature;
    }

    public Module getModule_image() { return this.module_image; }
    public void setModule_image(Module module_image)
    {
        this.module_image= module_image;
    }

    public Module getModule_text()
    {
        return this.module_text;
    }
    public void setModule_text(Module module_text)
    {
        this.module_text= module_text;
    }

    public float[] getSimilarity()
    {
        return this.Similarity;
    }
    public void setSimilarity(float[] Similarity)
    {
        this.Similarity= Similarity;
    }

    @Override
    public void onCreate()
    {
        super.onCreate();
        ImageSet=new ArrayList<>();
    }
}
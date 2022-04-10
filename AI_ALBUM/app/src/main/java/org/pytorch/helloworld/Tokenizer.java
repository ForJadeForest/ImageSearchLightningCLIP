package org.pytorch.helloworld;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Tokenizer {

    static TokenTool tokenTool;
    //TokenTool tokenTool;
    static Map<Integer, Character> encode_map;
    static List<String[]> merges;
    static List<String> vocab;
    static Map<String, Integer> encoder;
    static Map<String[], Integer> bpe_ranks;
    static Map<String, String> cache;
    static String pat;

    /*
    public Tokenizer(String filename) throws FileNotFoundException {
        //构建字典
        tokenTool=new TokenTool();
        File file = new File(filename);
        Scanner s = new Scanner(file);
        encode_map = new HashMap<>();
        merges=new ArrayList<>();
        vocab=new ArrayList<>();
        encoder=new HashMap<>();
        bpe_ranks=new HashMap<>();
        cache=new HashMap<>();

        while(s.hasNext())
        {
            merges.add(s.nextLine().split(" "));

        }
        merges=merges.subList(1,49152-256-2+1);
        encode_map=tokenTool.bytes_to_unicode();
        Generate_vocab();

        for(int i=0;i<vocab.size();i++)
        {
            encoder.put(vocab.get(i),i);
        }
        for(int i=0;i<merges.size();i++)
        {
            bpe_ranks.put(merges.get(i),i);
        }
        cache.put("<|startoftext|>","<|startoftext|>");
        cache.put("<|endoftext|>","<|endoftext|>");
    }
    */
    public Tokenizer(String filename) throws FileNotFoundException {
        //构建字典
        tokenTool=new TokenTool();
        File file = new File(filename);
        Scanner s = new Scanner(file);
        encode_map = new HashMap<>();
        merges=new ArrayList<>();
        vocab=new ArrayList<>();
        encoder=new HashMap<>();
        bpe_ranks=new HashMap<>();
        cache=new HashMap<>();

        while(s.hasNext())
        {
            vocab.add(s.nextLine());
            //merges.add(s.nextLine().split(" "));

        }
        //merges=merges.subList(1,49152-256-2+1);
        //encode_map=tokenTool.bytes_to_unicode();
        //Generate_vocab();

        for(int i=0;i<vocab.size();i++)
        {
            encoder.put(vocab.get(i),i);
        }
        /*
        for(int i=0;i<merges.size();i++)
        {
            bpe_ranks.put(merges.get(i),i);
        }

        cache.put("<|startoftext|>","<|startoftext|>");
        cache.put("<|endoftext|>","<|endoftext|>");*/
    }
    /*
    void bpe()
    {

    }
    */
    void Generate_vocab()
    {
        List<Character> t_vocab= new ArrayList<Character>(encode_map.values());
        List<String> tt_vocab= new ArrayList<String>();

        for(Character i : t_vocab)
        {
            vocab.add(String.valueOf(i));
        }
        for(String i : vocab)
        {
            tt_vocab.add(i+"</w>");
        }
        vocab.addAll(tt_vocab);
        for(String[] i : merges)
        {
            String t=new String();
            for(String j: i)
            {
                t=t+j;
            }
            vocab.add(t);
        }
        vocab.add("<|startoftext|>");
        vocab.add("<|endoftext|>");
    }

    static int[] encode(String text)
    {
        text = tokenTool.whitespace_clean(text).toLowerCase();
        String[] texts=text.split(" ");
        int [] bpe_token=new int[texts.length];
        for(int i=0;i<texts.length;i++)
        {
            //int t_token=encoder.getOrDefault(texts[i]+"</w>",-1);
            int t_token=encoder.get(texts[i]+"</w>");
            bpe_token[i]=t_token;
        }
        return bpe_token;
    }
}

class TokenTool{
    Map<Integer, Character> bytes_to_unicode()
    {
        Map<Integer, Character> encode_map=new HashMap<>();
        List<Integer> bs = new ArrayList<Integer>();
        for(int i=(int)('!');i<=(int)('~');i++)
        {
            bs.add(i);
        }
        for(int i=(int)('¡');i<=(int)('¬');i++)
        {
            bs.add(i);
        }
        for(int i=(int)('®');i<=(int)('ÿ');i++)
        {
            bs.add(i);
        }
        List<Integer> cs=new ArrayList<>();
        for(Integer i:bs)
        {
            cs.add(i);
        }
        int n=0;
        for(int i=0;i<256;i++)
        {
            if(!bs.contains(i))
            {
                bs.add(i);
                cs.add(256+n);
                n+=1;
            }
        }
        List<Character> Cs = new ArrayList<Character>();
        for(int i=0;i<cs.size();i++)
        {
            Cs.add((char)((int)(cs.get(i))));
        }
        for(int i=0;i<Cs.size();i++)
        {
            encode_map.put(bs.get(i),Cs.get(i));
        }
        return encode_map;
    }
    /*
    def get_pairs(word):
            """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
            for char in word[1:]:
            pairs.add((prev_char, char))
    prev_char = char
    return pairs


    def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
            return text.strip()

*/
    String whitespace_clean(String text)
    {
        text.replaceAll("\\s+"," ");
        return text;
    }


}

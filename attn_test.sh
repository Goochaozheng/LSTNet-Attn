for attn in scaled_dot cosine multihead
do
    python main.py --gpu=0 --horizon=3 --model=attn --data=traffic --attn_score=$attn
done
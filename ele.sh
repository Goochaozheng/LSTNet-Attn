for h in 3 6 12 24
do
    python main.py --gpu=0 --horizon=$h --model=skip --data=electricity --output_fun=Linear
done

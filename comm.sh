for c in 4 8 32 64 100
do
    echo "############################################"
    echo "hidCNN=$c"
    python main.py --gpu=0 --horizon=3 --data=commodity --window=14 --highway_window=7 --model=normal --batch_size=16 --hidCNN=$c --hidRNN=$c
done

for i in {G..J}; do
    for j in {A..K}; do
        python mol2_reader.py -i ./$i$j -o ./ghose_filtered/ &
    done
#    wait
done

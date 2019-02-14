for i in `seq 1 100`
do
    dot -Tpng iris-dtree$i.dot -o treeRamd$i.png
done

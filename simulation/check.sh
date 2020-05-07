./crun-omp -g ./data/g-180x160-uniA.gph -r ./data/r-180x160-d35-epi.rats -u b -n 20 -t 1 -i 1 -F t1 -q
./crun-omp -g ./data/g-180x160-uniA.gph -r ./data/r-180x160-d35-epi.rats -u b -n 20 -t 8 -i 1 -F t2 -q
python check.py t1_node_count.txt t2_node_count.txt

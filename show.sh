#num=$(ls -l | grep task | wc -l)
num=$(grep "START" $1/status.txt | wc -l)
#num=$(echo $num-1|bc)
for (( i=0; i<$num; ++i )); do
  cat $1/task-${i}.txt | tail -n 1 
done

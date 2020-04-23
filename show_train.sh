#num=$(ls -l | grep task | wc -l)
num=$(grep "START TASK" $1/alchemist.txt | wc -l)
#num=$(echo $num-1|bc)
for (( i=0; i<$num; ++i )); do
  cat $1/task-${i}.txt | grep accuracy | tail -n 1 
#  echo ""
done

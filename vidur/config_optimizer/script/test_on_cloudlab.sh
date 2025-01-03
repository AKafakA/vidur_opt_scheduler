#parallel-ssh -i -h vidur/config_optimizer/script/cl_host "sudo apt update && sudo apt install -y python3-pip"
#parallel-ssh -i -t 0 -h vidur/config_optimizer/script/cl_host "git clone https://github.com/AKafakA/vidur_opt_scheduler.git && cd vidur_opt_scheduler && git checkout exp && pip install -r requirements.txt"
parallel-ssh -i -t 0 -h vidur/config_optimizer/script/cl_host "cd vidur_opt_scheduler && git pull"
#parallel-ssh -t 0 -h vidur/config_optimizer/script/cl_host "pip install pyinstrument"
mapfile -t ary < <(sed 's/.*/&/' "./vidur/config_optimizer/script/cl_host")

counter=0
for filename in vidur/config_optimizer/config_explorer/config/experiment_config/global_*.yml; do
    echo "Running on ${ary[counter]}"
    parallel-ssh -i --host ${ary[counter]} "pkill -f vidur & rm -rf vidur_opt_scheduler/simulation_output"
    parallel-ssh --host ${ary[counter]} "cd vidur_opt_scheduler && nohup sh vidur/config_optimizer/script/run_simulation.sh ${filename} &"
    counter=$((counter+1))
done




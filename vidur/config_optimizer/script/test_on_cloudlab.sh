parallel-ssh -h vidur/config_optimizer/script/cl_host "sudo apt update && sudo apt install -y python3-pip"
parallel-ssh -t 0 -h vidur/config_optimizer/script/cl_host "git clone https://github.com/AKafakA/vidur_opt_scheduler.git && cd vidur_opt_scheduler && git checkout exp && pip install -r requirements.txt"
parallel-ssh -t 0 -h vidur/config_optimizer/script/cl_host "cd vidur_opt_scheduler && git pull"
mapfile -t ary < <(sed 's/.*/"&"/' "./vidur/config_optimizer/script/cl_host")

counter=0
for filename in vidur/config_optimizer/config_explorer/config/experiment_config/global_*.yml; do
    parallel-ssh -t 0 --host ${ary[counter]} "cd vidur_opt_scheduler &&  export PYTHONPATH=. && nohup python vidur/config_optimizer/config_explorer/main.py --config-path ${filename} --num-threads 10 --output-dir ./simulation_output &"
    counter=$((counter+1))
done




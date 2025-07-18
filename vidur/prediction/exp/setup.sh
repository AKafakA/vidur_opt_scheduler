parallel-ssh -t 0 -h vidur/prediction/config/hosts "sudo apt update && sudo apt full-upgrade -y"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "sudo apt install -y python3-pip python3-venv ccache"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "pip3 install -U pip==25.0.1"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "sudo cp /var/cuda-repo-ubuntu2004-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-6 && sudo apt-get install -y nvidia-open"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "echo 'export PATH=$PATH:/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "sudo nvidia-smi -mig 0"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "git clone https://github.com/AKafakA/vllm.git && cd vllm && sudo VLLM_USE_PRECOMPILED=1 pip install --editable ."
parallel-ssh -t 0 -h vidur/prediction/config/hosts "git clone https://github.com/AKafakA/vidur_opt_scheduler.git && cd vidur_opt_scheduler && git checkout single_predictor_evaluation  && pip install -r requirements.txt"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "rm -r ~/cuda-repo-*.deb"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126"
parallel-ssh -t 0 -h vidur/prediction/config/hosts "pip install flashinfer-python==0.2.5 triton==3.2.0"
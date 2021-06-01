import os
import argparse
import time

import os
import time
import random
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        sleep_time = random.random()
        time.sleep(sleep_time)#in order to avoid same directory collision#gen file before submitting jobs
        if not os.path.exists(path):
            os.makedirs(path)
        print(path + " created")
        return True
    else:
        print (path+' existed')
        return False
def write_slurm_sh(id,command_line, queue_name="learnfair",nodes=1,
                   gpu_per_node=8,wall_time=3*24*60,username="wang3702",CPU_PER_GPU=10):
    """
    Args:
        id: running id
        command_line: command line
        outlog_path: saving path
    Returns:

    """
    import time
    import datetime
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    dependency_handler_path = os.path.join(os.getcwd(),"src")
    dependency_handler_path = os.path.join(dependency_handler_path,"handler.txt")
    run_path = os.path.join(os.getcwd(),"log")
    mkdir(run_path)
    run_path = os.path.abspath(run_path)
    batch_file = os.path.join(run_path,"slurm_job_"+str(id)+".sh")
    output_path = os.path.join(run_path,"output_"+str(id)+"_"+str(formatted_today+now)+".log")
    error_path = os.path.join(run_path,"error_"+str(id)+"_"+str(formatted_today+now)+".log")
    with open(batch_file,"w") as file:
        file.write("#!/bin/sh\n")
        file.write("#SBATCH --job-name=%s\n"%id)
        file.write("#SBATCH --output=%s\n"%output_path)
        file.write("#SBATCH --error=%s\n"%error_path)
        file.write("#SBATCH --partition=%s\n"%queue_name)
        file.write("#SBATCH --signal=USR1@600\n")
        file.write("#SBATCH --nodes=%d\n"%nodes )
        file.write("#SBATCH --ntasks-per-node=%d\n"%gpu_per_node)
        file.write("#SBATCH --gpus=%d\n"%(nodes*gpu_per_node))
        file.write("#SBATCH --gpus-per-node=%d\n" % (gpu_per_node))
        file.write("#SBATCH --cpus-per-task=%d\n"%(CPU_PER_GPU))
        file.write("#SBATCH --time=%d\n"%wall_time)
        file.write("#SBATCH --mail-user=%s@fb.com\n"%username)
        file.write("#SBATCH --mail-type=FAIL\n")
        file.write("#SBATCH --mail-type=end \n")
        file.write('#SBATCH --constraint="volta"\n')
        report_info ="%s job failed; \t"%id
        report_info += "log path: %s; \t"%output_path
        report_info += "error record path: %s\t"%error_path
        report_info += "command line path: %s\t"%batch_file
        file.write('#SBATCH --comment="%s"\n'%(report_info))
        with open(dependency_handler_path,'r') as rfile:
            line = rfile.readline()
            while line:
                file.write(line)
                line = rfile.readline()

        file.write("bash /private/home/wang3702/.bashrc\n")
        #file.write("/private/home/wang3702/anaconda3/bin/conda init\n")
        file.write("/private/home/wang3702/anaconda3/bin/conda activate base\n")
        file.write("master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}\n")
        file.write('dist_url="tcp://"\n')
        file.write("dist_url+=$master_node\n")
        file.write("dist_url+=:40000\n")
        file.write(command_line+"\n")
        file.write("wait $!\n")
        file.write("set +x \n")
        file.write("echo ..::Job Finished, but No, AGI is to BE Solved::.. \n")
        # signal that job is finished
    os.system('sbatch ' + batch_file)

dump_path= os.path.join(os.getcwd(),"swav_dump_100")
mkdir(dump_path)
import time
import datetime
queue_name = "dev"
today = datetime.date.today()
formatted_today = today.strftime('%y%m%d')
now = time.strftime("%H:%M:%S")
dump_path = os.path.join(dump_path, formatted_today + now)
mkdir(dump_path)
command_line = "python -u main_swav.py --data_path imagenet --nmb_crops 2 6 --size_crops 224 96 " \
               "--min_scale_crops 0.14 0.05 --max_scale_crops 1. 0.14 --crops_for_assign 0 1 "\
                "--temperature 0.1  --epsilon 0.05  --sinkhorn_iterations 3  --feat_dim 128  " \
               "--nmb_prototypes 3000 --queue_length 3840  --epoch_queue_starts 15  --epochs 100 --batch_size 32 " \
               "--base_lr 0.6 --final_lr 0.0006 --freeze_prototypes_niters 5005  --wd 0.000001 --warmup_epochs 0  " \
               "--dist_url $dist_url --arch resnet50 --use_fp16 true  --sync_bn pytorch --dump_path %s " \
               ""%dump_path
write_slurm_sh("swav_baseline_100", command_line, queue_name)
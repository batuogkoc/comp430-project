srun -N 1 -n 8 --mem=64G --qos=users --partition=ai -t 1-00:00:00 --gres=gpu:1 --pty bash

import subprocess
import logging
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import torch


def run_and_log(cmd_arr, log_file):
    logging.debug(' '.join(cmd_arr))
    with open(log_file, "w") as log:
        p = subprocess.Popen(cmd_arr, universal_newlines=True, stdout=log, stderr=log)
        p.wait()

    # return content of the log file
    with open(log_file, 'r') as log:
        return log.readlines()
train_sets = [
    'courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow', 'office',
    'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains',
]

def main(args):
    output_path = Path(args.output_path)
    experiment_name = args.experiment_name
    checkpoint_name = 'last.ckpt' if (args.epoch_id is None) else 'epoch={}.ckpt'.format(args.epoch_id)
    ETH_BIN_PATH = Path(args.eth3d_binary_path)
    GT_PLY_PATH = Path(args.eth3d_gt_path)

    # obtain output experiment path
    dataset = 'ETH3DHR'
    output_exp_path = output_path / experiment_name /  checkpoint_name / dataset

    # output_ply_path = output_exp_path / 'ETH3DHR'
    output_ply_path = output_exp_path  / 'pointclouds'
    output_eval_path = output_exp_path / 'evals'
    output_xls_path = output_eval_path / 'results.xlsx'
    set_ids = train_sets
    com_data = {}
    acc_data = {}
    f1_data = {}
    # tolerances = [0.01,0.02,0.05,0.1,0.2,0.5]
    tolerances = [0.02,0.05]
    ts = [str(t) for t in tolerances]

    for set_id in tqdm(set_ids, desc="processing set"):    
        set_ply_path = output_ply_path / (set_id +  '.ply')
        gt_mlp_path = GT_PLY_PATH / set_id / 'dslr_scan_eval' / 'scan_alignment.mlp'
        eval_ply_path = output_eval_path / set_id / set_id
        eval_log_path = output_eval_path / (set_id + '_log.txt')

        eval_ply_path.parent.mkdir(parents=True, exist_ok=True)

        cmd_arr = [
            str(ETH_BIN_PATH),
            "--tolerances", ','.join(ts),
             "--reconstruction_ply_path", str(set_ply_path),
             "--ground_truth_mlp_path", str(gt_mlp_path),
             "--completeness_cloud_output_path", str(eval_ply_path) + '.completeness',
             "--accuracy_cloud_output_path", str(eval_ply_path) + '.accuracy',
        ]
        log_lines = run_and_log(cmd_arr, eval_log_path)
        tol_line = log_lines[-4]
        com_line = log_lines[-3]
        acc_line = log_lines[-2]
        f1_line = log_lines[-1]
        # tolerances = [float(v) for v in tol_line.split()][1:]
        completeness = [float(v) for v in com_line.split()[1:]]
        accuracies = [float(v) for v in acc_line.split()[1:]]
        f1s= [float(v) for v in f1_line.split()[1:]]

        # create a set_id + tolerance / completeness / accuracy / f1 graph
        com_data[set_id] = completeness
        acc_data[set_id] = accuracies
        f1_data[set_id] = f1s
    coms = pd.DataFrame.from_dict(com_data, orient='index', columns=tolerances)
    accs = pd.DataFrame.from_dict(acc_data, orient='index', columns=tolerances)
    f1s = pd.DataFrame.from_dict(f1_data, orient='index', columns=tolerances)
    writer = pd.ExcelWriter(output_xls_path)
    coms.to_excel(writer, sheet_name='completeness')
    accs.to_excel(writer, sheet_name='accuracies')
    f1s.to_excel(writer, sheet_name='f1')
    writer.save()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--eth3d_binary_path', type=str)
    parser.add_argument('--eth3d_gt_path', type=str)
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--epoch_id', type=int, default=None)

    args = parser.parse_args()

    main(args)
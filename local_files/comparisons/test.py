import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GraphCast forecasts against datasets like ERA5/IMERG")

    # Evaluation range
    parser.add_argument('--eval_start', type=str, required=True,
                        help='Start date for evaluation period (format: YYYY-MM-DD)')
    parser.add_argument('--eval_end', type=str, required=True,
                        help='End date for evaluation period (format: YYYY-MM-DD)')

    # Dataset choice
    parser.add_argument('--eval_dataset_choice', type=str, choices=['era5', 'imerg', 'imd', 'stations'], default='era5',
                        help='Dataset choice for evaluation')
    
    parser.add_argument('--vars_to_eval', type=str, required=True, help='Which variables to evaluate')

    # Data paths
    parser.add_argument('--eval_data_path', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_cache/',
                        help='Path to Eval dataset')
    
    parser.add_argument('--params_path_old', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/gc_weights/origs/graphcast_1_13.npz',
                        help='Path to original GraphCast parameters (.npz)')
    
    parser.add_argument('--params_path_new1', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig.npz',
                        help='Path to fine-tuned GraphCast parameters (.npz)')


    parser.add_argument('--params_path_new2', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig.npz',
                        help='Path to fine-tuned GraphCast parameters (.npz)')
    

    parser.add_argument('--norms_dir', type=str, default='/Datastorage/saptarishi.dhanuka_asp25/gc_norms/',
                        help='Directory containing normalization .nc files')

    # Output paths
    parser.add_argument('--output_pred_old_dir', type=str,
                        default='/Datastorage/saptarishi.dhanuka_asp25/preds_dir/',
                        help='Output path for original GraphCast predictions')
    parser.add_argument('--output_pred_finetuned_dir', type=str,
                        default='/Datastorage/saptarishi.dhanuka_asp25/preds_dir/',
                        help='Output path for fine-tuned GraphCast predictions')
    parser.add_argument('--plots_dir', type=str, default='plots/evals',
                        help='Directory to save eval plots')

    parser.add_argument('--plot_timesteps', type=int, default=4,
                        help='Number of timesteps to generate plots for')

    # Spatial extent
    parser.add_argument('--latmin', type=float, default=6, help='Minimum latitude for plotting')
    parser.add_argument('--latmax', type=float, default=38, help='Maximum latitude for plotting')
    parser.add_argument('--lonmin', type=float, default=68, help='Minimum longitude for plotting')
    parser.add_argument('--lonmax', type=float, default=98, help='Maximum longitude for plotting')

    parser.add_argument(
    "--params_paths",
    nargs="+",   # allows multiple values
    type=str,
    required=True,
    help="List of new model parameter paths to evaluate."
)


    return parser.parse_args()



args = parse_args()

print(args.params_paths)

new_params_list = []
for path in args.params_paths:
    print(path)
    with open(path, "rb") as f:
        # ckpt_new = checkpoint.load(f, graphcast.CheckPoint)
        new_params_list.append((path))

print(new_params_list)

# python test.py --eval_start "2014-08-01" --eval_end "2014-12-30" --eval_dataset_choice "imerg" --vars_to_eval "total_precipitation_6hr" --params_paths ${BASE_PATH}graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_sharp_mask_expt7.npz ${BASE_PATH}finetuned/graphcast_1_13_orig_2014-06-01_2014-07-30_FORECAST28_dynamic_weighing_india_mask_expt3_good.npz

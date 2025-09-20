import papermill as pm

pm.execute_notebook(
    'eval_forecast.ipynb',
    'eval_forecast_out.ipynb',
    parameters=dict(
        # Parameters
        eval_start = "2024-08-01",
        eval_end = "2024-09-15",
        dataset_choice = "imerg",
        eval_vars = "total_precipitation_6hr",
        apath = "/Datastorage/saptarishi.dhanuka_asp25/era5_data/era5_cache/",
        params_path_old = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/origs/graphcast_1_13.npz',

        params_path_new = '/Datastorage/saptarishi.dhanuka_asp25/gc_weights/graphcast_1_13_orig_2024-06-01_2024-07-30_FORECAST28.npz',
        norms_dir = '/Datastorage/saptarishi.dhanuka_asp25/gc_norms/',
        plots_dir = 'plots/evals',
        latmin = 6,
        latmax = 38,
        lonmin = 35,
        lonmax = 65,
        plot_timesteps = 7,
        output_pred_old_dir = '/Datastorage/saptarishi.dhanuka_asp25/preds_dir/',
        output_pred_finetuned_dir = '/Datastorage/saptarishi.dhanuka_asp25/preds_dir/'
    )
    )

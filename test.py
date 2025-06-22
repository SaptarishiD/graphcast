from google.cloud import storage
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
blob = gcs_bucket.blob("dataset/source-era5_date-2022-01-01_res-0.25_levels-13_steps-12.nc")
print("Downloading")
blob.download_to_filename("/Datastorage/saptarishi.dhanuka_asp25/source-era5_date-2022-01-01_res-0.25_levels-13_steps-12.nc")

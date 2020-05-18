FOR %%i IN (JAK1.csv, JAK2.csv, JAK3.csv) DO ^
python -m slgnn.data_processing.jakfp_dataset -p data\JAK\%%i
Download the data from https://drive.google.com/open?id=1tjXmyce0bfdaTiUk4BzrG1oV2ZQLmITH or https://storage.cloud.google.com/ppzt_ml2_project/pku-autonomous-driving%20(1).zip and unzip them into /Data/ folder

The structure of the data should looks like:

|--\Data

|----\train_images 

|----\train_masks 

|----\test_images

|----\test_masks

|----\car_models

|----\car_models_json

|----\camera

|--\EfficientDet.py

|--\EfficientDet.py

|--\train.csv

|--\sample-submission.csv

To execute the script and reproduce the results, simply run the EfficientDet.py


There are some additional repos we studied, and some of our code are borrowed or modified from their work. The related works are folked in the following repos:

https://github.com/yle8458/CenterNet 

https://github.com/yle8458/EfficientNet-PyTorch

import os
import shutil

import subprocess

from pathlib import Path
import zipfile


kaggleJson = "kaggle.json"
home = str(Path.home())
kagglePath = os.path.abspath(os.path.join(home , ".kaggle"))
keyjsonPath = os.path.abspath(os.path.join(home , "Downloads" , kaggleJson))


if not os.path.exists(kagglePath):
    print("Created Directory {}".format(kagglePath))
    os.mkdir(kagglePath)

if os.path.exists(keyjsonPath) \
 and not os.path.exists(os.path.abspath(os.path.join(kagglePath, kaggleJson))):
    print("Copying Key file to Kaggle Path")
    shutil.copy(keyjsonPath, dst=kagglePath)
else:
    if not os.path.exists(keyjsonPath):
        print("Please Download the Kaggle API json from kaggle.com/[USERNAME]/account")

competitionName = "widsdatathon2020"
dataPath = os.path.abspath(os.path.join(os.path.dirname(__file__) , "..","Data" ))
zipPath = os.path.abspath(os.path.join(dataPath , '{}.zip'.format(competitionName)))
os.chdir(dataPath)
if not os.path.exists(zipPath):
    subprocess.call(["kaggle" , "competitions", "download" , "-c" ,competitionName])


if True not in [f.find(".csv") > -1 for f in os.listdir(dataPath)]:
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall(dataPath)

print("All Set!")

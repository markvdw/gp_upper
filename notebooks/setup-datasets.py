# Copyright 2017 Mark van der Wilk.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.from __future__ import print_function

# Download and process all datasets
from __future__ import print_function

import os
import shutil
import sys
import zipfile

import numpy as np
import requests
from scipy.io import savemat

basepath = os.path.dirname(os.path.realpath(__file__))

datasets_store_dir = os.path.join(basepath, ".")
download_target_folder = os.path.join(basepath, 'raw_download/')
process_temp_folder = os.path.join(basepath, 'proc_temp/')

required_directories = [datasets_store_dir, download_target_folder, process_temp_folder]

download_urls = {"snelson": "http://www.gatsby.ucl.ac.uk/~snelson/SPGP_dist.zip", }


def download_file(url, target_dir='.'):
    local_filename = "%s/%s" % (target_dir, url.split('/')[-1])
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                # f.flush() commented by recommendation from J.F.Sebastian
    return local_filename


def process_snelson():
    print("Processing snelson")
    zf = zipfile.ZipFile("%s/SPGP_dist.zip" % download_target_folder)
    zf.extractall(process_temp_folder)
    X = np.loadtxt('%s/SPGP_dist/train_inputs' % process_temp_folder)[:, None]
    Y = np.loadtxt('%s/SPGP_dist/train_outputs' % process_temp_folder)[:, None]
    savemat("%s/snelson1d.mat" % datasets_store_dir,
            {'X': X, 'Y': Y, 'tX': X, 'tY': Y, 'name': "snelson1d",
             'description': "Snelson 1d classic dataset",
             "url": download_urls["snelson"]})


def setup_datasets():
    for dir in required_directories:
        if not os.path.exists(dir):
            os.mkdir(dir)

    print("Downloading files... This may take a while...")
    for url in download_urls.values():
        local_filename = url.split("/")[-1]
        print("%-*s" % (40, local_filename), end="... ")
        sys.stdout.flush()
        if not os.path.exists("%s/%s" % (download_target_folder, local_filename)):
            download_file(url, download_target_folder)
        else:
            print("Skipping.", end="")
        print("")
    print("")

    print("Processing downloaded files...")
    process_snelson()

    shutil.rmtree(process_temp_folder)
    shutil.rmtree(download_target_folder)


if __name__ == "__main__":
    setup_datasets()

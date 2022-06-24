#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
json2dvideow.py
Convert JSON files of output Openpose 1.3 in .dat to Dvideow
Created: Jun 15 2020 | Update: Aug 02 2020
@authors: Luiz Henrique P. Vieira; Allan Pinto; Paulo R. P. Santiago
email: paulosantiago@usp.br
output <https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md>
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

def main(people):
    json_dat0 = data0['people'][people]['pose_keypoints_2d']
    ncoord = len(json_dat0[0::3])
    nfiles = len(json_files)

    x = np.zeros([nfiles, ncoord])
    y = np.zeros([nfiles, ncoord])

    for i in range(nfiles):
        json_data = open(json_files[i])
        # print(json_data)
        data = json.load(json_data)
        try:
            json_dat = data['people'][people]['pose_keypoints_2d']
            x[i,:] = json_dat[0::3]
            y[i,:] = json_dat[1::3]
        except:
            x[i,:] = [1 for _ in range(ncoord)]
            y[i,:] = [1 for _ in range(ncoord)]

    xy = np.zeros([nfiles, 2*ncoord])
    xy[:,0::2] = x
    xy[:,1::2] = y 

    linframes = np.arange(nfiles)
    matdat = np.column_stack((linframes,xy))
    name2save = '../dvideow_people_'+str(people)+'.dat'
    np.savetxt(name2save, matdat, delimiter=' ', fmt='%d'+' %.6f'*ncoord*2)
    return matdat

if __name__ == "__main__":
    dir_atual = sys.argv[1]
    os.chdir(dir_atual)
    json_files = sorted(glob.glob('*.json'))
    json_data0 = open(json_files[0]) # para extrair o numero de marcadores
    data0 = json.load(json_data0)
    numpeople = len(data0['people'])
    
    for i in range(numpeople):
        main(people=i)
    
    body_25 = {0: 'Nose',
        1: 'Neck',
        2: 'RShoulder',
        3: 'RElbow',
        4: 'RWrist',
        5: 'LShoulder',
        6: 'LElbow',
        7: 'LWrist',
        8: 'MidHip',
        9: 'RHip',
        10: 'RKnee',
        11: 'RAnkle',
        12: 'LHip',
        13: 'LKnee',
        14: 'LAnkle',
        15: 'REye',
        16: 'LEye',
        17: 'REar',
        18: 'LEar',
        19: 'LBigToe',
        20: 'LSmallToe',
        21: 'LHeel',
        22: 'RBigToe',
        23: 'RSmallToe',
        24: 'RHeel',
        25: 'Background'}
    
    print('\nResult for BODY_25 (25 body parts consisting of COCO + foot):')
    print(body_25)
    print('\n********* Congratulations all done! ********')
    print('The files with {} peoples was saved!\n'.format(numpeople))

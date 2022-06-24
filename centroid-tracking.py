# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.cluster import KMeans
from common import retrieve_fnames, safe_create_dir


def compute_moments(points):

    try:
        M = cv2.moments(points)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        pass
    except Exception as e:
        print("WARNING:", e)
        cx, cy = points.max(axis=0)

    return (cx,cy)


def compute_cg(dat, anthropometric_model='0', show_frame=0):
    # dat = np.genfromtxt(filename, delimiter=' ')
    # dat = np.loadtxt('dvideow_people_0.dat', delimiter=' ', skiprows=0)

    dat = dat[:,1:]

    cg_cab = dat[:,0:2] #Nose
    olho_d = dat[:,30:32] #REye
    olho_e = dat[:,32:34] #LEye
    orelha_d = dat[:,34:36] #REar
    orelha_e = dat[:,36:38] #LEar
    neck = dat[:,2:4] #pescoco
    midhip = dat[:,16:18] #MidHip
    ombro_d = dat[:,4:6] #RShoulder
    ombro_e = dat[:,10:12] #LShoulder
    cotovelo_d = dat[:,6:8] #RElbow
    cotovelo_e = dat[:,12:14] #LElbow
    punho_d = dat[:,8:10] #RWrist
    punho_e = dat[:,14:16] #LWrist
    quadril_d = dat[:,18:20] #RHip
    quadril_e = dat[:,24:26] #LHip
    joelho_d = dat[:,20:22] #RKnee
    joelho_e = dat[:,26:28] #LKnee
    tornozelo_d = dat[:,22:24] #RAnkle
    tornozelo_e = dat[:,28:30] #LAnkle
    calcanhar_d = dat[:,48:50] #RHell
    calcanhar_e = dat[:,42:44] #LHell
    bigtoe_d = dat[:,44:46] #right big toe
    bigtoe_e = dat[:,38:40] #left big toe
    smalltoe_d = dat[:,46:48] #left small toe
    smalltoe_e = dat[:,40:42] #right small toe
    pontape_d = (bigtoe_d + smalltoe_d) / 2 #mean point between RBigToe and RSmallToe
    pontape_e = (bigtoe_e + smalltoe_e) / 2 #mean point between LBigToe and LSmallToe

    cg_perc = {'cabeca':[1, 1], 'tronco':[.431, .3782], 'braco':[.5772, .5754],
                 'antebraco':[.4574, .4559], 'coxa':[.4095, .3612], 
                 'perna':[.4395, .4352], 'pe':[.4415, .4014]}

    if anthropometric_model == '1':
        sexo = 1
        modelo = 'mulher'
    else:
        sexo = 0
        modelo = 'homen'
    
    print(f'Modelo Antropométrico: {modelo}')
    # Tronco
    cg_tronco = neck + cg_perc['tronco'][sexo] * (midhip - neck) #tronco
    #  braco 
    cg_braco_e = ombro_e + cg_perc['braco'][sexo] * (cotovelo_e - ombro_e) #braco esquerdo
    cg_braco_d = ombro_d + cg_perc['braco'][sexo] * (cotovelo_d - ombro_d) #braco direito
    # antebraco
    cg_antebraco_e = cotovelo_e + cg_perc['antebraco'][sexo] * (punho_e - cotovelo_e) #antebraco esquerdo
    cg_antebraco_d = cotovelo_d + cg_perc['antebraco'][sexo] * (punho_d - cotovelo_d) #antebraco direito
    # coxa
    cg_coxa_e = quadril_e + cg_perc['coxa'][sexo] * (joelho_e - quadril_e) #coxa esquerda
    cg_coxa_d = quadril_d + cg_perc['coxa'][sexo] * (joelho_d - quadril_d) #coxa direita
    # perna
    cg_perna_e = joelho_e + cg_perc['perna'][sexo] * (tornozelo_e - joelho_e)
    cg_perna_d = joelho_d + cg_perc['perna'][sexo] * (tornozelo_d - joelho_d)
    # pe
    cg_pe_e = calcanhar_e + cg_perc['pe'][sexo] * (pontape_e - calcanhar_e)
    cg_pe_d = calcanhar_d + cg_perc['pe'][sexo] * (pontape_d - calcanhar_d)
    
    # Center of mass / center of gravity 
    cg_total = ((0.081 * cg_cab)   + (0.497 * cg_tronco)  + (0.028 * cg_braco_d)   + (0.028 * cg_braco_e) + 
    (0.022 * cg_antebraco_d) + (0.022 * cg_antebraco_e) + (0.1 * cg_coxa_d) + (0.1 * cg_coxa_e) +
    (0.047 * cg_perna_d) + (0.047 * cg_perna_e) + (0.014 * cg_pe_d)   + (0.014 * cg_pe_e)) / 1
    
    if show_frame > 0:
        segcab = np.stack((cg_cab[show_frame,:], neck[show_frame,:]), axis=0)
        segtronco = np.stack((ombro_e[show_frame,:], ombro_d[show_frame,:], quadril_d[show_frame,:], quadril_e[show_frame,:], ombro_e[show_frame,:]))    
        segbracod = np.stack((ombro_d[show_frame,:], cotovelo_d[show_frame,:]))
        segbracoe = np.stack((ombro_e[show_frame,:], cotovelo_e[show_frame,:]))
        segabracod = np.stack((cotovelo_d[show_frame,:], punho_d[show_frame,:]))
        segabracoe = np.stack((cotovelo_e[show_frame,:], punho_e[show_frame,:]))
        segcoxad = np.stack((quadril_d[show_frame,:], joelho_d[show_frame,:]))
        segcoxae = np.stack((quadril_e[show_frame,:], joelho_e[show_frame,:]))
        segpernad = np.stack((joelho_d[show_frame,:], tornozelo_d[show_frame,:]))
        segpernae = np.stack((joelho_e[show_frame,:], tornozelo_e[show_frame,:]))
        segped = np.stack((calcanhar_d[show_frame,:], pontape_d[show_frame,:]))
        segpee = np.stack((calcanhar_e[show_frame,:], pontape_e[show_frame,:]))   
        fig1, ax = plt.subplots()
        ax.plot(segcab[:,0], segcab[:,1])
        ax.plot(segtronco[:,0], segtronco[:,1])
        ax.plot(segbracod[:,0], segbracod[:,1])
        ax.plot(segbracoe[:,0], segbracoe[:,1])
        ax.plot(segabracod[:,0], segabracod[:,1])
        ax.plot(segabracoe[:,0], segabracoe[:,1])
        ax.plot(segcoxad[:,0], segcoxad[:,1])
        ax.plot(segcoxae[:,0], segcoxae[:,1])
        ax.plot(segpernad[:,0], segpernad[:,1])
        ax.plot(segpernae[:,0], segpernae[:,1])
        ax.plot(segped[:,0], segped[:,1])
        ax.plot(segpee[:,0], segpee[:,1])
        ax.plot(cg_cab[show_frame,0], cg_cab[show_frame,1], 'k.', markersize=18)
        ax.plot(cg_braco_d[show_frame,0], cg_braco_d[show_frame,1], 'k.', markersize=18)
        ax.plot(cg_braco_e[show_frame,0], cg_braco_e[show_frame,1], 'k.', markersize=18)
        ax.plot(cg_antebraco_d[show_frame,0], cg_antebraco_d[show_frame,1], 'k.', markersize=18)
        ax.plot(cg_antebraco_e[show_frame,0], cg_antebraco_e[show_frame,1], 'k.', markersize=18)
        ax.plot(cg_tronco[show_frame,0], cg_tronco[show_frame,1], 'k.', markersize=18)
        ax.plot(cg_coxa_d[show_frame,0], cg_coxa_d[show_frame,1], 'k.', markersize=18)
        ax.plot(cg_coxa_e[show_frame,0], cg_coxa_e[show_frame,1], 'k.', markersize=18)
        ax.plot(cg_perna_d[show_frame,0], cg_perna_d[show_frame,1], 'k.', markersize=18)
        ax.plot(cg_perna_e[show_frame,0], cg_perna_e[show_frame,1], 'k.', markersize=18)
        ax.plot(cg_pe_d[show_frame,0], cg_pe_d[show_frame,1], 'k.', markersize=18)
        ax.plot(cg_pe_e[show_frame,0], cg_pe_e[show_frame,1], 'k.', markersize=18)
        ax.plot(midhip[show_frame,0], midhip[show_frame,1], 'k.', markersize=18)
        # ax.plot(neck[show_frame,0], neck[show_frame,1], 'k.', markersize=18)
        # ax.plot(cg_cab[show_frame,0],cg_cab[show_frame,1], 'k.', markersize=18)
        ax.plot(olho_d[show_frame,0], olho_d[show_frame,1], 'g.', markersize=6)
        ax.plot(olho_e[show_frame,0], olho_e[show_frame,1], 'g.', markersize=6)
        # ax.plot(orelha_d[show_frame,0], orelha_d[show_frame,1], 'y.', markersize=6)
        # ax.plot(orelha_e[show_frame,0], orelha_e[show_frame,1], 'y.', markersize=6)
        ax.plot(cg_total[show_frame,0], cg_total[show_frame,1], 'k*', markersize=14)
        ax.set_box_aspect(2)
        plt.title(f'Frame number = {show_frame}; Modelo Antropométrico: {modelo}')
        plt.show()
    
    return cg_total


def draw_points(input_img, points, center_of_mass):
    try:
        img = cv2.imread(input_img, cv2.IMREAD_COLOR)
        # img = np.zeros((max_coords[1], max_coords[0], 3), np.uint8)+255
        # max_coords = points.max(axis=0)+10
        cv2.circle(img, tuple(center_of_mass), 3, (255,0,0), -1)
        for point in points:
            cv2.circle(img, tuple(point), 3, (0,0,255), -1)
        pass
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()
    return img


def averaged_points(all_data, fnames_imgs):
    all_cms = []
    for k, data in enumerate(all_data):
        cms = []
        for coords, input_img in tqdm(zip(data.values, fnames_imgs)):
            points = np.reshape(coords, (-1,2))
            points = np.round(points).astype(np.int)
            # import pdb; pdb.set_trace()
            cm = points.mean(axis=0).astype(np.int)
            cms += [cm]

        all_cms += [cms]

    return np.array(all_cms)


def cg_points(all_data, fnames_imgs, anthropometric_model, show_frame):
    all_cms = []
    for k, data in enumerate(all_data):
        all_cms += [compute_cg(data.values, anthropometric_model=anthropometric_model, show_frame=show_frame)]

    return np.array(all_cms)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dat_file_path', type=str, default='', help='(default: %(default)s)')
    parser.add_argument('--frames', type=str, default='', help='(default: %(default)s)')
    parser.add_argument('--output_path', type=str, default='', help='(default: %(default)s)')
    parser.add_argument('--anthropometric_model', type=str, default='', help='(default: %(default)s)')
    parser.add_argument('--show_frame', type=int, default=0, help='(default: %(default)s)')
    parser.add_argument('--method', type=str, default='averaged_points', choices=['averaged_points', 'cg_points'], help='(default: %(default)s)')

    args = parser.parse_args()

    fnames = retrieve_fnames(args.dat_file_path, [".dat"])

    fnames_imgs = retrieve_fnames(args.frames, [".png"])

    fnames = [fname for fname in fnames if args.method not in fname]

    all_data = []
    for fname in fnames:
        all_data += [pd.read_csv(fname, delimiter=" ", header=None).drop(0, axis=1)]

    if "averaged_points" in args.method:
        all_cms = averaged_points(all_data, fnames_imgs)
    elif "cg_points" in args.method:
        all_cms = cg_points(all_data, fnames_imgs, args.anthropometric_model, args.show_frame)
    else:
        raise("Please choose a valid method!")

    n_clusters = len(fnames)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.concatenate(all_cms))
    print("# clusters", n_clusters)
    persons_idxs = []
    for c in range(n_clusters):
        persons_idxs += [np.where(kmeans.labels_ == c)[0]]
        print("-- Centroid {}: {} poses:".format(c,len(persons_idxs[-1])))

    all_data = pd.concat(all_data)
    for k, fname in enumerate(fnames):
        name, ext = os.path.splitext(fname)
        out_fname = "{}_{}{}".format(name, args.method, ext)
        new_index = np.arange(len(persons_idxs[k])).reshape(-1,1)
        new_data = all_data.iloc[persons_idxs[k]].sort_index().values
        pd.DataFrame(np.concatenate((new_index,new_data), axis=1)).to_csv(out_fname, sep=" ", header=False, index=False)
        
        for coords, input_img in tqdm(zip(new_data, fnames_imgs)):
            points = np.reshape(coords, (-1,2))
            points = np.round(points).astype(np.int)
            cm = points.mean(axis=0).astype(np.int)
            img = draw_points(input_img, points, cm)
            out_fname = os.path.join(os.path.dirname(fnames[k]), "frames_{}_{}".format(args.method, k), os.path.basename(input_img))
            safe_create_dir(os.path.dirname(out_fname))
            cv2.imwrite(out_fname, img)

if __name__ == '__main__':
    main()

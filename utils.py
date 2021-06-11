# my commonly used functions

import numpy as np
import random
import itertools
import warnings

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_hex
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import scipy
from scipy.cluster.vq import kmeans

from sklearn.metrics import confusion_matrix
from sklearn import neighbors

import subprocess
import os
import shutil

# Standardizing classification schemes
Types = np.array([1,2,3,4,5])
Oxys = np.array([-2, 0, 2])
Categories = ['(1) Ali.', '(1) Aro.', '(2) Ali.', '(2) Aro.', '(3) Ali.', '(3) Aro.',
	'(4) Ali.', '(4) Aro.', '(5) Ali.', '(5) Aro.']

batch_size = 50

# Standardizing colors
COLORMAP = plt.cm.viridis
COLORS = list(COLORMAP(np.arange(5)/4))
COLORS[4] = COLORS[4] - (25/255,25/255,25/255,0)
COLORS[3] = COLORS[3] - (10/255,10/255,10/255,0)
COLORS[2] = COLORS[2] - (30/255,30/255,35/255,0)
COLORS[1] = COLORS[1] + (0/255,20/255,40/255,0)

# TEST set metadata
TEST_OXY = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,3])
TEST_TYPE = np.array([1,1,1,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5])
TEST_CAT = np.array([1,1,1,1,2,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,9,9,9,10,10,10])


def read_tddft_spectrum_file(path):
    return np.loadtxt(path).T


def read_tddft_transitions_file(path):
    return np.loadtxt(path).T  


def get_transition_sum(filename):
    x, y = read_tddft_transitions_file(filename)
    kbeta_trans = []
    kalpha_trans = []
    for i in range(len(x)):
        if x[i] > 2445 and x[i] < 2480:
            kbeta_trans.append(y[i])
        if x[i] > 2300 and x[i] < 2320:
            kalpha_trans.append(y[i])
    return np.sum(kbeta_trans)/np.sum(kalpha_trans)


def get_oxy_from_type(t):
    if t==1 or t==2 or t==3:
        return -2
    if t == 4:
        return 0
    if t== 5:
        return 2    
    return -1


def get_conj(Conj, NoConj, c):
    if c in Conj:
        return 1
    if c in NoConj:
        return 0
    return -1


def get_text_in_file(infile):
    # Read in the file
    with open(infile, 'r') as f:
        data = f.read()
    return data


def get_category(t, conj):
    category = -1
    # type t not conj vs conj
    base = 2*t - 1
    if conj == 0:
        category = base
    if conj == -1:
        print("Compound has no conjugated category")
        return base
    if conj == 1:
        category = base + 1
    return category


def get_XANES(Types, TYPEdir):
    Data = []
    iterator = 1
    for t in Types:
        try:
            file = open(f'{TYPEdir}Type{t}/Type{t}.txt', 'r')
            DATdir = f"Data/Type{t}/"

            Conj = get_text_in_file(f'{TYPEdir}Type{t}/Conj.txt')
            NoConj = get_text_in_file(f'{TYPEdir}Type{t}/NoConj.txt')

            if os.path.exists(f'{TYPEdir}Type{t}/Mixed.txt'):
                Mixed = get_text_in_file(f'{TYPEdir}Type{t}/Mixed.txt')
                mixed = True
            else:
                mixed = False

            for line in file:
                compound = line.replace('\n','')
                
                spectrum_XANES = read_tddft_spectrum_file(f'Data/Type{t}/{compound}/XANES/{compound}.processedspectrum')
                trans = read_tddft_spectrum_file(f'Data/Type{t}/{compound}/XANES/{compound}.dat')
                
                conj = -1
                
                conj = get_conj(Conj, NoConj, compound)

                if mixed:
                    if compound in Mixed:
                        mix = 1
                    else:
                        mix = 0
                else:
                    mix = -1

                cat = get_category(t, conj)
                temp_dict = {'name': compound, 'Type': t, 'XANES': spectrum_XANES, 'Transitions': np.flip(trans, axis=1),
                            'oxy': get_oxy_from_type(t), 'conj':conj, 'category': cat, 'Mixed': mix}
                Data.append(temp_dict)
                print(f'{iterator}\r', end="")
                iterator += 1
        finally:
            file.close()
    return Data


def get_XES(Types, TYPEdir):
    Data = []
    iterator = 1
    for t in Types:
        try:
            file = open(f'{TYPEdir}Type{t}/Type{t}.txt', 'r')
            DATdir = f"Data/Type{t}/"

            Conj = get_text_in_file(f'{TYPEdir}Type{t}/Conj.txt')
            NoConj = get_text_in_file(f'{TYPEdir}Type{t}/NoConj.txt')

            for line in file:
                compound = line.replace('\n','')
                spectrum_XES = read_tddft_spectrum_file(f'Data/Type{t}/{compound}/XES/{compound}.processedspectrum')
                trans = read_tddft_spectrum_file(f'Data/Type{t}/{compound}/XES/{compound}.dat')
                conj = -1
                conj = get_conj(Conj, NoConj, compound)
                tsum = get_transition_sum(f'Data/Type{t}/{compound}/XES/{compound}.dat')

                cat = get_category(t, conj)
                temp_dict = {'name': compound, 'Type': t, 'XES': spectrum_XES, 'Transitions': np.flip(trans, axis=1),
                            'oxy': get_oxy_from_type(t), 'conj':conj, 'category': cat, 'Tsum': tsum}
                Data.append(temp_dict)
                print(f'{iterator}\r', end="")
                iterator += 1
        finally:
            file.close()
    return Data


def get_TEST_Data(compound_list, directory='XES'):
    Data = []
    iterator = 1
    for compound in compound_list:
        spectrum = read_tddft_spectrum_file(f'Data/TEST/{compound}/{directory}/{compound}.processedspectrum')
        transitions = read_tddft_spectrum_file(f'Data/TEST/{compound}/{directory}/{compound}.dat')         
        temp_dict = {'Name': compound, 'Spectra': spectrum, 'Transitions': np.flip(transitions, axis=1)}
        Data.append(temp_dict)
        print(f'{iterator}\r', end="")
        iterator += 1
    return Data


def get_Property(Dict_list, myproperty, normalize=True):
    temp = []
    for ele in Dict_list:
        if myproperty is 'Type':
            type_num = ele['Type']
            if normalize:
                one_hot_encoded_label = np.zeros(5)
                one_hot_encoded_label[type_num-1] = 1
                temp.append(one_hot_encoded_label)
            else:
                temp.append(type_num)
        elif myproperty is 'oxy':
            oxy = ele['oxy']
            if normalize:
                one_hot_encoded_label = np.zeros(3)
                for i in range(len(Oxys)):
                    if oxy == Oxys[i]:
                         one_hot_encoded_label[i] = 1
                temp.append(one_hot_encoded_label)
            else:
                temp.append(oxy) 
        elif myproperty is 'category':
            cat = ele['category']
            one_hot_encoded_label = np.zeros(10)
            one_hot_encoded_label[cat-1] = 1
            temp.append(one_hot_encoded_label)
        else:
            temp.append(ele[myproperty])            
    if myproperty is 'Tsum' and normalize: # normalize
        temp -= min(temp)
        temp = temp/max(temp)
    return temp


def Spagetti_plot(Data, energy, scaling, mode='VtC-XES'):
    
    if mode == 'VtC-XES':
        MIN, MAX = 50,900
        mode = 'XES'
    elif mode == 'XANES':
        MIN, MAX = 0, -1
    
    fig, ax = plt.subplots(figsize=(8,18))
    base = 1
    
    Peaks = np.ones((5))
    
    i = 0
            
    for ele in Data:
        t = ele["Type"]
        y = ele[mode][1]
        y = y / scaling
        
        if mode == 'XES':
            if t == 2:
                y = y*0.8
        elif mode == 'XANES':
            if t == 1:
                y = y*1.4
            if t == 2:
                y = y*2
            if t == 3:
                y = y*1.6
            if t == 5:
                y = y*0.8
      
        x = energy[MIN:MAX]
        y = y[MIN:MAX]

        if t == 5:
            if i%3 == 0:
                plt.plot(x, y + t-1, '-', c=COLORS[t-1], alpha=0.1)
        elif t == 1:
            if i%5 != 0:
                plt.plot(x, y + t-1, '-', c=COLORS[t-1], alpha=0.1)
        else:
            plt.plot(x, y + t-1, '-', c=COLORS[t-1], alpha=0.1)

        i += 1
        
    plt.plot([2470],[4.8],'wo')
                
    if mode == 'XES':
        mode = 'VtC-XES'
        plt.xticks([2450,2455,2460,2465,2470,2475], ['',2455,'',2465,'',2475], fontsize=26)
    elif mode == 'XANES':
        plt.xticks([2470,2475,2480,2485,2490,2495], [2470,'',2480,'',2490,''], fontsize=26)
    
    plt.title(f"{mode} Spectra", fontsize=30)
    plt.xlabel('Energy (eV)', fontsize=26)
    plt.yticks([],fontsize=22)
    ax.tick_params(direction='in', width=2, length=8)

    plt.show()


def one_hot_to_num(arr):
    return np.argmax(arr, axis=1)+1


def get_Spectrum(Dict_list, name, mode='XES'):
    for ele in Dict_list:
        if ele['name'] == name:
            return ele[mode]


def plot_spectrum(spectrum, compound, ticks=True, figsize=(12,8)):
    x, y = spectrum

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y, 'k-', linewidth=1, label=compound)

    plt.title(compound, fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if not ticks:
    	plt.axis('off')
    plt.show()


def plot_spectrum_and_trans(transitions, spectrum, name, figsize=(12,8), label=None, ylab=False, emp=None):
    x, y = spectrum
    xs, ys = transitions

    fig, ax = plt.subplots(figsize=figsize)

    if label is None:
        label = name

    ax.plot(x, y, 'k-', label=label)
    
    markerline, stemlines, baseline = ax.stem(xs, ys, linefmt='-', markerfmt='o', use_line_collection=True)
    
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, 'linewidth', 1, color='gray', alpha=0.5)
    plt.setp(markerline, 'markersize', 4, color='gray', alpha=0.5)

    if emp is not None: 
        markerline, stemlines, baseline = \
            ax.stem(xs[emp], ys[emp], linefmt='-', markerfmt='o', use_line_collection=True)    
        plt.setp(baseline, visible=False)
        plt.setp(stemlines, 'linewidth', 2, color='r', alpha=1)
        plt.setp(markerline, 'markersize', 4, color='r', alpha=1)

    plt.tick_params(labelsize=16)
    plt.title(name, fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=16)
    if ylab:
        plt.ylabel('Intensity (arb. units)', fontsize=16)
    else:
        plt.yticks([])

    ax.tick_params(direction='in', width=2, length=8)

    if label is not None:
        legend = ax.legend([label], fontsize=24, handlelength=0, handletextpad=0, fancybox=True,)
        for item in legend.legendHandles:
            item.set_visible(False)
    plt.show()


def index_from_array(array, val):
    dif = (array - val)**2
    return np.argmin(dif)


def get_avg(mylist):
    return np.sum(np.array(mylist))/len(mylist)


def get_std(mylist):
    mu = get_avg(mylist)
    return np.sqrt(np.sum((np.array(mylist) - mu)**2)/len(mylist))


def get_avg_and_std(x, y):
    # x = Type categories of data
    # y = Data itself
    Avg = np.zeros(5)
    Std = np.zeros(5)
    for t in Types:
        temp = []
        for i in range(len(x)):
            if x[i] == t:
                temp.append(y[i])
        Avg[t-1] = get_avg(temp)
        Std[t-1] = get_std(temp)
    return Avg, Std


def plot_Type_vs_Transition_Sum(Dict_list):

    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.array(get_Property(Dict_list, 'Type', normalize=False))
    y = np.array(get_Property(Dict_list, 'Tsum', normalize=False))
    
    avg, std = get_avg_and_std(x, y)
    print(f'avg: {avg}')
    print(f'std: {std}')
    cmap = ListedColormap(COLORS)
    plt.scatter(x, y, c=x, alpha=0.4, cmap=cmap)
    plt.errorbar(Types, avg, yerr=std, fmt='k', label="Averaged transition integral")    
    
    cbar = plt.colorbar(ticks=Types)
    cbar.set_label('Type', fontsize=26)
    cbar.ax.tick_params(labelsize=22)
    cbar.set_alpha(1)
    cbar.draw_all()

    plt.xlabel('Type', fontsize=20)
    plt.ylabel(f'Transition Integral Ratio (arb. units)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.tick_params(direction='in', width=2, length=8)
    plt.ylim(0, 0.015)

    plt.show()


def make_plot(Data, hist_data, TYPE, max_range, minimax, show_conj=False):
    y, y2 = [], []
    for ele in Data:
        if ele['Type'] == TYPE:
            tsum = ele['Tsum']
            if max_range=='all' or tsum < float(max_range):
                if show_conj:
                    if ele['conj'] == 0:
                        y.append(tsum)
                    else:
                        y2.append(tsum)
                else:
                    y.append(tsum)

                if tsum > minimax[1][0]:
                    minimax[1] = [ele['Tsum'], ele['name']]
                if tsum < minimax[0][0]:
                    minimax[0] = [ele['Tsum'], ele['name']]

    hist_data.append(y)
    if show_conj:
        hist_data.append(y2)
    return hist_data, minimax


def create_transition_v_type_hist(Data, max_range="all", chosen_type=None, show_heavy=False,
                                  show_minimax=False, show_conj=False, amide=False, n_bins = 30):
    
    fig, ax = plt.subplots(1, 1, figsize=(12,8))

    hist_data = []
    minimax = [[np.Infinity, ''],[0, '']]

    Colors = list(COLORMAP(np.arange(1,11)/9))
    # type 1
    Colors[0] = COLORS[0].copy()
    Colors[1] = '#9F5F80'
    # type 2
    Colors[2] = '#03506F'
    Colors[3] = COLORS[1].copy() + (60/255,80/255,90/255,0.)
    # type 3
    Colors[4] = '#DB6400' 
    Colors[5] = '#ffba93' 
    # type 4
    Colors[6] = '#2b3016'
    Colors[7] = COLORS[3].copy()
    # type 5
    Colors[8] = '#ac3501'
    Colors[9] = COLORS[4]
    
    if chosen_type == None:
        for t in Types: 
            hist_data, minimax = make_plot(Data, hist_data, t, max_range,
                minimax, show_conj=show_conj)
        if not show_conj:
            Colors = COLORS.copy()
        label = [f'Type {t}' for t in Types]
    else:
        # type given
        hist_data, minimax = make_plot(Data, hist_data, chosen_type, max_range, 
                minimax, show_conj=show_conj)
        if show_conj:
            base = 2*chosen_type - 2
            temp = [Colors[base], Colors[base+1]]
            Colors= temp.copy()
            label = ['Not conjugated', 'Conjugated'] # not conjugated vs conjugated
        else:
            Colors = COLORS[chosen_type-1]
            label = f'Type {chosen_type}'
    
    if show_minimax:
        print(minimax)

    if len(hist_data) > 10:
        hist_data = [hist_data]  
    
    # plot
    ax.hist(hist_data, bins=n_bins, label=label, histtype='bar', color=Colors)

    plt.xlabel(f'Transition Integral Ratio (arb. units)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.tick_params(direction='in', width=2, length=8)

    ax.legend(fontsize=20)

    plt.show()


def get_prediction(ANN, x_test, y_test):  
    prediction = ANN.predict(x_test)
    predicted_labels = one_hot_to_num(prediction)
    true_labels = one_hot_to_num(y_test)  
    return np.array(true_labels), np.array(predicted_labels)


def get_type_eles(Data, t):
    temp = []
    for ele in Data:
        if ele['Type'] == t:
            temp.append(ele)
    return temp


def plot_confusion_matrix(cm, classes, PROPERTY, cmap, title='Confusion matrix'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    fontsize=20
    if PROPERTY == 'oxy':
    	title = title + ': Oxidation'
    elif PROPERTY == 'Type':
    	title = title + ': Type'
    elif PROPERTY == 'category':
    	title = title + ': Aromaticity'
    	fontsize=24
    plt.title(title, fontsize=fontsize)
    cbar = plt.colorbar()
    cbar.set_label('', fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=fontsize-4)
    plt.yticks(tick_marks, classes, fontsize=fontsize-4)
    
    fmt = '.2f'
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=fontsize)
    plt.xlabel('Predicted label', fontsize=fontsize)


def normalizeCM(cm):
    C = np.zeros(cm.shape)
    for i in range(cm.shape[0]):
        c = cm[i,:]/np.sum(cm[i,:])
        for j in range(cm.shape[1]):
            C[i][j] = c[j]
    return C


def make_confusion_matrix(true, pred, labels, PROPERTY, cmap=plt.cm.Blues):
    cm = confusion_matrix(true, pred)        
    CM = normalizeCM(cm)

    np.set_printoptions(precision=2)
    if PROPERTY == 'category':
    	plt.figure(figsize=(12,9))
    else:
    	plt.figure(figsize=(8,6))
    plot_confusion_matrix(CM, labels, PROPERTY, cmap=cmap)
    plt.show()


def get_xtrain_names(Data, index_shuffle, train_size):
    NAMES = get_Property(Data, 'name')
    Xshuffle = []
    for index in index_shuffle:
        Xshuffle.append(NAMES[index])
    x_train_names = Xshuffle[:train_size]
    return x_train_names


def get_xtest_names(Data, index_shuffle, train_size):
    NAMES = get_Property(Data, 'name')
    Xshuffle = []
    for index in index_shuffle:
        Xshuffle.append(NAMES[index])
    x_test_names = Xshuffle[train_size:]
    return x_test_names


def get_dict_from_names(Data, x_names):
    temp = []
    for ele in Data:
        for name in x_names:
            if ele['name'] == name:
                temp.append(ele.copy())
    return temp


def shuffle_xy(X, Y):
    # random shuffle of data
    index_shuffle = np.arange(len(X))
    random.shuffle(index_shuffle)
    Xshuffle = []
    Yshuffle = []
    for index in index_shuffle:
        Xshuffle.append(X[index].copy())
        Yshuffle.append(Y[index].copy())
    Xshuffle = np.array(Xshuffle)
    Yshuffle = np.array(Yshuffle)
    return Xshuffle, Yshuffle


def show_combined_loss(history, xes_losses):
    
    xes_loss, xes_val_loss = xes_losses

    fig, ax = plt.subplots(figsize=(12, 8))
    fontsize=24
    color=COLORS[2]
    
    # summarize history for loss
    plt.plot([6], [6], c='w', alpha=1.0, label='    XANES')
    plt.plot(history.history['loss'], '--', c=COLORS[1], linewidth=2, label='Training Loss')
    plt.plot(history.history['val_loss'], '-', c=COLORS[1], linewidth=4, label='Validation Loss')
    plt.plot([6], [6], c='w', alpha=1.0, label='\n   VtC-XES')
    
    plt.plot(xes_loss, '--', c=COLORS[3], linewidth=2, label='Training Loss')
    plt.plot(xes_val_loss, '-', c=COLORS[3], linewidth=4, label='Validation Loss')

    ax.set_ylim(bottom=185, top=275)
    
    plt.ylabel('VAE Loss', fontsize=fontsize+4)
    plt.xlabel('Epoch', fontsize=fontsize+4)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(-1,np.min([len(xes_loss), len(history.history['loss'])]))
    ax.tick_params(direction='out', width=2, length=8)
    ax.tick_params(direction='out', which='minor', width=1, length=5)
    
    plt.legend(fontsize=fontsize)
    plt.show()


def show_loss(history, model):
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fontsize=16
    color=COLORS[2]
    
    # summarize history for loss
    plt.plot(np.log(history.history['loss']), '--', c='k', linewidth=2)
    plt.plot(np.log(history.history['val_loss']), '-', c=color, linewidth=4)
    plt.title(f'{model} VAE Model Loss', fontsize=fontsize+4)
    plt.ylabel('Log Loss', fontsize=fontsize)
    plt.xlabel('Epoch', fontsize=fontsize)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.legend(['Train', 'Validation'], fontsize=fontsize)
    plt.show()


def stack_plot(x, ys, names, title, space=1., figsize=(12,8), leg=2, fontsize=16,
               ncol=1, method=None, legend_font=None):

    n = len(ys)
    if n is 5 or n < 5:
        Colors = COLORS
    else:
        Colors = COLORMAP(np.arange(n)/(n+1))
    
    fig, ax = plt.subplots(figsize=figsize)

    ymax = n*space
    for i in range(n):
        plt.plot(x, ys[i] + ymax - i*space, '-', c=Colors[i], label=names[i])

    plt.title(f"{title}", fontsize=24)
    plt.xlabel('Energy (eV)', fontsize=fontsize+4)
    # plt.ylabel('Intensity (arb. units)', fontsize=16)

    if method is None:
        plt.xticks(fontsize=fontsize+2)
    elif method == 'XES':
        plt.xticks([2450,2455,2460,2465,2470,2475], [2450,'',2460,'',2470,''], fontsize=26)
    elif method == 'XANES':
    	plt.xticks([2470,2475,2480,2485,2490,2495], [2470,'',2480,'',2490,''], fontsize=26)
    plt.yticks([],fontsize=fontsize)
    ax.tick_params(direction='in', width=2, length=8)


    if legend_font is None:
        legend_font = fontsize+2
    if leg != 0:
        plt.legend(fontsize=legend_font, loc=leg, ncol=ncol)
    plt.show()


def plot_stacked_spectrum(energy, x1, x2, name1, name2, title='Spectra', space=1, loc=None, method=None):
    
    x1 = x1/np.max(x1)
    x2 = x2/np.max(x2)
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    if len(x1) is not 1000:
        energy = energy[:len(x1)]

    plt.plot(energy, x2+space, '-', c=COLORS[1], label=name2)
    plt.plot(energy, x1, '-', c=COLORS[3], label=name1)

    plt.title(f"{title}", fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=24)
    if method is None:
    	plt.xticks(fontsize=24)
    elif method == 'XES':
    	plt.xticks([2450,2455,2460,2465,2470,2475], [2450,'',2460,'',2470,''], fontsize=24)
    elif method == 'XANES':
    	plt.xticks([2465,2470,2475,2480,2485,2490,2495,2500,2505], [2465,'','',2480,'','',2495,'',''], fontsize=24)
    plt.yticks([],fontsize=14)
    ax.tick_params(direction='in', width=2, length=8)

    if loc is not None:
    	plt.legend(loc=loc, fontsize=22)
    else:
    	plt.legend(fontsize=26)
    plt.show()

def stacked_trans(energy, in1, in2, names, title=None, space=0,
                  loc=None, method=None, figsize=(12,8)):
    
    x1, trans1 = in1
    x2, trans2 = in2
    name1, name2 = names

    x1 = x1/np.max(x1)
    x2 = x2/np.max(x2)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(x1) is not 1000:
        energy = energy[:len(x1)]

    plt.plot(energy, x2+space, '-', c=COLORS[0], label=name2)
    markerline, stemlines, baseline = ax.stem(trans2[0], trans2[1],
        linefmt='-', markerfmt='o', use_line_collection=True)
    
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, 'linewidth', 1, color=COLORS[0])
    plt.setp(markerline, 'markersize', 4, color=COLORS[0])


    plt.plot(energy, x1, '-', c=COLORS[2], label=name1)
    markerline, stemlines, baseline = ax.stem(trans1[0], trans1[1],
        linefmt='-', markerfmt='o', use_line_collection=True)
    
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, 'linewidth', 1, color=COLORS[2])
    plt.setp(markerline, 'markersize', 4, color=COLORS[2])

    if title is not None:
    	plt.title(f"{title}", fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=24)
    if method is None:
    	plt.xticks(fontsize=24)
    elif method == 'XES':
    	plt.xticks([2450,2455,2460,2465,2470,2475], [2450,'',2460,'',2470,''], fontsize=24)
    elif method == 'XANES':
    	plt.xticks([2465,2470,2475,2480,2485,2490,2495,2500,2505], [2465,'','',2480,'','',2495,'',''], fontsize=24)
    plt.yticks([],fontsize=14)
    ax.tick_params(direction='in', width=2, length=8)

    if loc is not None:
    	plt.legend(loc=loc, fontsize=22)
    else:
    	plt.legend(fontsize=26)
    plt.show()


def plot_in_v_out(energy, vae, x_predict, test_names,
	   x_range=(0,-1), space=0.75, c=2, figsize=(12,8), method='XES'):
    
    vae_out = vae.predict(x_predict, batch_size=1)
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, x in enumerate(x_predict):  
        x = x[x_range[0]: x_range[1]]
        x2 = vae_out[i,:][x_range[0]: x_range[1]]
        if method == 'XANES':
            if i in [0,1,2]:
                x = x*1.1
                x2 = x2*1.1
            else:
                x = x*0.8
                x2 = x2*0.8
        elif method == 'XES':
            if i == 1:
                x2 = x2*0.7
                x = x*0.7
        plt.plot(energy[x_range[0]: x_range[1]], x+space*i, 'k--', label="Input")
        plt.plot(energy[x_range[0]: x_range[1]], x2+space*i,
                 '-', c=COLORS[c], label="Decoded")

    plt.plot([2470], [3.7], 'wo')

    plt.xlabel('Energy (eV)', fontsize=24)
    if method is None:
    	plt.xticks(fontsize=24)
    elif method == 'XES':
    	plt.xticks([2450,2455,2460,2465,2470,2475], [2450,'',2460,'',2470,''], fontsize=24)
    elif method == 'XANES':
    	plt.xticks([2470,2475,2480,2485,2490,2495], [2470,'',2480,'',2490,''], fontsize=24)
    plt.yticks([],fontsize=14)
    ax.tick_params(direction='in', width=2, length=8)

    plt.legend(["Input", "Decoded"], fontsize=24)
    plt.show()


def type_spagetti(energy, names, Data, mode, X, Type=3, space=0, figsize=(8, 6), alpha=0.1):
    
    Colors = list(COLORMAP(np.arange(1,11)/9))
            
    # type 1
    Colors[0] = COLORS[0].copy()
    Colors[1] = '#9F5F80'
    # type 2
    Colors[2] = '#03506F'
    Colors[3] = COLORS[1].copy() + (60/255,75/255,75/255,0.)
    # type 3
    Colors[4] = '#DB6400'
    Colors[5] = '#ffba93'
    # type 4
    Colors[6] = '#2b3016' 
    Colors[7] = COLORS[3].copy()
    # type 5
    Colors[8] = '#ac3501'
    Colors[9] = COLORS[4]
    
    mn, mx = 10, -200
    
    Aro = np.zeros_like(energy)
    n_aro = 0
    Ali = np.zeros_like(energy)
    n_ali = 0
       
    # Spaghetti 
    fig, ax = plt.subplots(figsize=figsize)
    
    for i in range(len(Data)):
        ele = Data[i]        
        if ele["Type"] == Type:
            base = 2*ele["Type"] - 2
            if ele['conj'] == 1:                
                plt.plot(energy[mn:mx], X[i][mn:mx] + space, '-', c=Colors[base+1], alpha=alpha) 
                Aro += X[i]
                n_aro += 1
            else:
                plt.plot(energy[mn:mx], X[i][mn:mx], '-', c=Colors[base], alpha=alpha)
                Ali += X[i]
                n_ali += 1
                
    plt.xlabel('Energy (eV)', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks([],fontsize=14)
    ax.tick_params(direction='in', width=2, length=8)
    
    ali = mpatches.Patch(color=Colors[base], label=f'Aliphatic')
    aro = mpatches.Patch(color=Colors[base+1], label=f'Aromatic')
    title = mpatches.Patch(color='w', label='XANES')
    
    legend = ax.legend(handles=[title, aro, ali], fancybox=True, fontsize=20)
    plt.title(f"Type {Type}", fontsize=24)
    plt.show()
    
    # Residual
    fig, ax = plt.subplots(figsize=figsize)
    
    Ali_avg = Ali/n_ali
    plt.plot(energy[mn:mx], Ali_avg[mn:mx], '-', c=Colors[base], linewidth=4)
    Aro_avg = Aro/n_aro
    plt.plot(energy[mn:mx], Aro_avg[mn:mx], '-', c=Colors[base+1], linewidth=4)
    
    Residual = Aro_avg - Ali_avg
    plt.plot(energy[mn:mx], Residual[mn:mx], '-', c='k', linewidth=3, alpha=0.8)
    
    plt.xlabel('Energy (eV)', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks([],fontsize=14)
    ax.tick_params(direction='in', width=2, length=8)
    
    ali = mpatches.Patch(color=Colors[base], label=f'Aliphatic')
    aro = mpatches.Patch(color=Colors[base+1], label=f'Aromatic')
    res = mpatches.Patch(linestyle='-', color='k', label=f'Residual')
    title = mpatches.Patch(color='w', label=mode)
    
    legend = ax.legend(handles=[title, aro, ali, res], fancybox=True, fontsize=20)
    plt.title(f"Type {Type}", fontsize=24)
    plt.show()


def bar_chart(Acc, Benchmarks, mode):
    x = ['VAE', 'PCA', 'FastICA', 'FA', 'NMF', 't-SNE']
    labels=['Oxidation','Type','Aromaticity']

    x_pos = np.array([i for i, _ in enumerate(x)])

    Colors = [COLORS[0],COLORS[2],COLORS[3]]

    fig = plt.figure(figsize=(16.5,8.5))
    ax1 = fig.add_subplot(20,1,(1,19))
    ax2 = fig.add_subplot(20,1,20)

    plt.subplots_adjust(hspace=0.3)

    for i in range(3):
        width=0.25
        ax1.bar(x_pos + width*i, Acc[i], width=width, color=Colors[i], label=labels[i])
        ax2.bar(x_pos + width*i, Acc[i], width=width, color=Colors[i], label=labels[i])


    plt.yticks(fontsize=22)
    plt.xticks(x_pos+0.25, x, fontsize=24)

    ax1.set_ylabel(f'Accuracy (%)', fontsize=24)
    ax1.tick_params(axis='y', direction='in', width=3, length=9)
    ax2.tick_params(axis='y', direction='in', width=3, length=9)

    ax2.tick_params(axis='x',direction='out', width=3, length=9)

    plt.setp(ax1.get_xticklabels(), Fontsize=22)
    plt.setp(ax1.get_yticklabels(), Fontsize=22)

    plt.setp(ax2.get_xticklabels(), Fontsize=22)
    plt.setp(ax2.get_xticklabels(), Fontsize=22)

    ax2.set_yticks([0])

    ax1.set_ylim(45, 100.5)
    ax2.set_ylim(0, 1) 

    ax1.axes.xaxis.set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    for i,bench in enumerate(Benchmarks):
        ax1.axhline(y=bench, color=Colors[i], linestyle='--', linewidth=3)

    ax1.plot(0,0, color='k', linestyle='--', linewidth=3, label='Benchmarks')

    ax1.legend(ncol=4,fontsize=24,loc=2,bbox_to_anchor=(0., 1.15))
    ax1.set_title(f'{mode}\n\n', fontsize=28)
    # plt.suptitle("XANES",fontsize=30)

    d = .01  # how big to make the diagonal lines in axes coordinates

    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=2)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    resize=22.5
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - resize*d, 1 + resize*d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - resize*d, 1 + resize*d), **kwargs)  # bottom-right diagonal

    plt.show()


def plot_dim_red(PROPERTY, X_red, Y, method, xloc=0.65, fontsize=28, black=False,
                 mode='VtC-XES', indices=None, special_label=None):
    
    cbar_vis=True
    
    Colors = list(COLORMAP(np.arange(1,11)/9))
    for i in range(10):
        Colors[i] = (128/255,128/255,128/255,0.2)
            
    # type 1
    Colors[0] = COLORS[0].copy()
    Colors[1] = '#9F5F80'
    # type 2
    Colors[2] = '#03506F'
    Colors[3] = COLORS[1].copy() + (60/255,75/255,75/255,0.)
    # type 3
    Colors[4] = '#DB6400' 
    Colors[5] = '#ffba93' 
    # type 4
    Colors[6] = '#2b3016' 
    Colors[7] = COLORS[3].copy()
    # type 5
    Colors[8] = '#ac3501'
    Colors[9] = COLORS[4]

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    
    if PROPERTY == 'oxy':
        Colors = [COLORS[0], COLORS[2], COLORS[4]]
        cmap = ListedColormap(Colors)
        plt.scatter(X_red[:, 0], X_red[:, 1], c=one_hot_to_num(Y)*2-4, cmap=cmap)
        ticks=Oxys
        label = 'Oxidation'
        loc = [-1.33, 0, 1.33]
        
    elif PROPERTY == 'category':
        
        label = "Category"
        
        Y = one_hot_to_num(Y)
        
        t1a = mpatches.Patch(color=Colors[0], label='Type 1')
        t1b = mpatches.Patch(color=Colors[1], label='Type 1')
        t2a = mpatches.Patch(color=Colors[2], label='Type 2')
        t2b = mpatches.Patch(color=Colors[3], label='Type 2')
        t3a = mpatches.Patch(color=Colors[4], label='Type 3')
        t3b = mpatches.Patch(color=Colors[5], label='Type 3')
        t4a = mpatches.Patch(color=Colors[6], label='Type 4')
        t4b = mpatches.Patch(color=Colors[7], label='Type 4')
        t5a = mpatches.Patch(color=Colors[8], label='Type 5')
        t5b = mpatches.Patch(color=Colors[9], label='Type 5')
        space = mpatches.Patch(color='w', label='')
        aro = mpatches.Patch(color='w', label='Aromatic')
        ali = mpatches.Patch(color='w', label='Aliphatic')
            
        fig.subplots_adjust(right=0.75)
        handles = [ali, t1a,t2a, t3a, t4a, t5a, space, aro, t1b, t2b, t3b, t4b, t5b]
        leg1 = plt.legend(handles=handles, ncol=1, fontsize=20, handletextpad=0.001,
                              bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.gca().add_artist(leg1)
        i = 0
        for patch in leg1.get_patches():
            patch.set_height(15)
            if i in [0, 7]:
                patch.set_width(0.1)
            else:
                patch.set_width(30)
            i += 1
        
        xy=(xloc,0.8)
        ax.annotate(f'{mode}:\n{method}',
                    xy=xy, xytext=xy,
                    xycoords='axes fraction',
                    textcoords='offset points',
                    size=fontsize)

        cbar_vis = False
        
        cmap = ListedColormap(Colors)
        plt.scatter(X_red[:, 0], X_red[:, 1], c=Y, cmap=cmap)
        
        if indices is not None:
            subset = X_red[indices]
            color = '#FF007F'
            plt.scatter(subset[:,0], subset[:,1], c=color)
            special = mpatches.Patch(color=color, label=f'{special_label}')
            leg2 = plt.legend(handles=[special], fontsize=20, handletextpad=0.01,
                              loc=2, borderaxespad=0., bbox_to_anchor=(1.01, 0.1))
            plt.gca().add_artist(leg2)
            for patch in leg2.get_patches():
                patch.set_height(15)
                patch.set_width(30)
        
    elif PROPERTY == 'Type':
        Colors = COLORS
        cmap = ListedColormap(Colors)
        plt.scatter(X_red[:, 0], X_red[:, 1], c=one_hot_to_num(Y), cmap=cmap)
        ticks=Types
        label = "Type"
        loc = [1.4, 2.2, 3, 3.8, 4.6]
        
    elif PROPERTY in Types: 
        base = 2*PROPERTY - 2
        subColors = [Colors[base], Colors[base+1]]
        cmap = ListedColormap(subColors)
        plt.scatter(X_red[:, 0], X_red[:, 1], c=Y, cmap=cmap)
        ticks=['Ali.','Aro.']
        label = f"Type {PROPERTY}"
        loc = [0.25,0.75]  
        
    if black:
        plt.scatter(X_red[:, 0], X_red[:, 1], c='k')
        cbar_vis=False
        
    if cbar_vis:
        cbaxes = fig.add_axes([0.92, 0.15, 0.033, 0.7])
        cbar = plt.colorbar(cax=cbaxes, ticks=ticks)
        cbar.set_label(label, fontsize=26)
        cbar.set_ticks(loc)
        cbar.ax.tick_params(labelsize=22)
        cbar.set_ticklabels(ticks)
        legend = ax.legend([method], handlelength=0, handletextpad=0, fancybox=True, fontsize=fontsize+12)
        for item in legend.legendHandles:
            item.set_visible(False)
    
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    plt.show()


def plot_dim_red_stack(PROPERTY, X_reds, Y, methods, fontsize=28, mode='VtC-XES'):
    
    n = len(X_reds)
    
    Colors = list(COLORMAP(np.arange(1,11)/9))
    for i in range(10):
        Colors[i] = (128/255,128/255,128/255,0.2)
        
    cbar_vis = True
            
    # type 1
    Colors[0] = COLORS[0].copy()
    Colors[1] = '#9F5F80'
    # type 2
    Colors[2] = '#03506F'
    Colors[3] = COLORS[1].copy() + (60/255,75/255,75/255,0.)
    # type 3
    Colors[4] = '#DB6400'
    Colors[5] = '#ffba93'
    # type 4
    Colors[6] = '#2b3016' 
    Colors[7] = COLORS[3].copy()
    # type 5
    Colors[8] = '#ac3501'
    Colors[9] = COLORS[4]

    fig, axes = plt.subplots(3,2, figsize=(10,15))
    
    if PROPERTY == 'oxy':
        Colors = [COLORS[0], COLORS[2], COLORS[4]]
        cmap = ListedColormap(Colors)
        
        index = 0
        for i, ax_tuple in enumerate(axes):
            for j, ax in enumerate(ax_tuple):
                plot = ax.scatter(X_reds[index][:, 0], X_reds[index][:, 1], c=one_hot_to_num(Y)*2-4, cmap=cmap)
                index += 1
        
        ticks=Oxys
        title = 'Oxidation'
        loc = [-1.33, 0, 1.33]
        label = title
        
    elif PROPERTY == 'category':
        
        title = "Category"
        label = title
            
        title = 'Aromatic'
            
        t1a = mpatches.Patch(color=Colors[0], label='Type 1')
        t1b = mpatches.Patch(color=Colors[1], label='Type 1')
        t2a = mpatches.Patch(color=Colors[2], label='Type 2')
        t2b = mpatches.Patch(color=Colors[3], label='Type 2')
        t3a = mpatches.Patch(color=Colors[4], label='Type 3')
        t3b = mpatches.Patch(color=Colors[5], label='Type 3')
        t4a = mpatches.Patch(color=Colors[6], label='Type 4')
        t4b = mpatches.Patch(color=Colors[7], label='Type 4')
        t5a = mpatches.Patch(color=Colors[8], label='Type 5')
        t5b = mpatches.Patch(color=Colors[9], label='Type 5')
        space = mpatches.Patch(color='w', label='')
        aro = mpatches.Patch(color='w', label='Aromatic')
        ali = mpatches.Patch(color='w', label='Aliphatic')
            
        handles = [ali, t1a,t2a, t3a, t4a, t5a, space, aro, t1b, t2b, t3b, t4b, t5b]
        leg1 = plt.legend(handles=handles, ncol=1, fontsize=20, handletextpad=0.001,
                              bbox_to_anchor=(1.03, 2.23), loc=2, borderaxespad=0.)
        plt.gca().add_artist(leg1)
        i = 0
        for patch in leg1.get_patches():
            patch.set_height(15)
            if i in [0, 7]:
                patch.set_width(0.1)
            else:
                patch.set_width(30)
            i += 1
            
        cbar_vis = False
        
        cmap = ListedColormap(Colors)
        
        index = 0
        for i, ax_tuple in enumerate(axes):
            for j, ax in enumerate(ax_tuple):
                plot = ax.scatter(X_reds[index][:, 0], X_reds[index][:, 1], c=one_hot_to_num(Y), cmap=cmap)
                index += 1
        
    elif PROPERTY == 'Type':
        Colors = COLORS
        cmap = ListedColormap(Colors)
        
        index = 0
        for i, ax_tuple in enumerate(axes):
            for j, ax in enumerate(ax_tuple):
                plot = ax.scatter(X_reds[index][:, 0], X_reds[index][:, 1], c=one_hot_to_num(Y)*2-4, cmap=cmap)
                index += 1
        
        ticks=Types
        title = "Type"
        loc = [-1.2, 0.4, 2., 3.6, 5.2]
        label = title
    
    index = 0
    for i, ax_tuple in enumerate(axes):
        for j, ax in enumerate(ax_tuple):
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            legend = ax.legend([methods[index]], handlelength=0, handletextpad=0, fancybox=True, fontsize=fontsize)
            for item in legend.legendHandles:
                item.set_visible(False)
            index += 1

    plt.subplots_adjust(hspace=0.0, wspace=0.)
    
    if cbar_vis:
        cbaxes = fig.add_axes([0.91, 0.2, 0.033, 0.6])
        cbar = plt.colorbar(plot, cax=cbaxes, ticks=ticks)
        cbar.set_label(label, fontsize=26)
        cbar.set_ticks(loc)
        cbar.ax.tick_params(labelsize=26)
        cbar.set_ticklabels(ticks)
    
    plt.suptitle(f'\n\n{mode}', fontsize=30)
    
    plt.show()


def train_KNN(x_train, y_train, n_neighbors):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(x_train, y_train)
    
    x_min, x_max = np.min(x_train[:,0]), np.max(x_train[:,0])
    y_min, y_max = np.min(x_train[:,1]), np.max(x_train[:,1])
    
    buffer = np.abs(x_max - x_min)*0.05
    h = (np.abs(x_max - x_min) + 2*buffer) / 100
    
    xx, yy = np.meshgrid(np.arange(x_min-buffer, x_max+buffer, h), np.arange(y_min-buffer, y_max+buffer, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return clf, xx, yy, Z


def normalize_train_test(train, test):
    Std = np.std(train, axis=0)
    Mean = np.mean(train, axis=0)
    train -= Mean
    train /= Std
    test -= Mean
    test /= Std 
    return train, test


def plot_KNN_stack(PROPERTY, X_red_train, X_red_test, Y, methods, fontsize=28, mode='VtC-XES',
                   n_neighbors=40):
    
    n = len(X_red_train)
    Y_train = one_hot_to_num(Y)
    
    if PROPERTY == 'oxy':
        Y_test = TEST_OXY
    elif PROPERTY == 'Type':
        Y_test = TEST_TYPE
    elif PROPERTY == 'category':
        Y_test = TEST_CAT
    
    Colors = list(COLORMAP(np.arange(1,11)/9))
    for i in range(10):
        Colors[i] = (128/255,128/255,128/255,0.2)
        
    cbar_vis = True
            
    # type 1
    Colors[0] = COLORS[0].copy()
    Colors[1] = '#9F5F80'
    # type 2
    Colors[2] = '#03506F'
    Colors[3] = COLORS[1].copy() + (60/255,75/255,75/255,0.)
    # type 3
    Colors[4] = '#DB6400'
    Colors[5] = '#ffba93'
    # type 4
    Colors[6] = '#2b3016' 
    Colors[7] = COLORS[3].copy()
    # type 5
    Colors[8] = '#ac3501'
    Colors[9] = COLORS[4]

    fig, axes = plt.subplots(3,2, figsize=(10,15))
    
    if PROPERTY == 'oxy':
        Colors = [COLORS[0], COLORS[2], COLORS[4]]
        cmap = ListedColormap(Colors)
        
        index = 0
        for i, ax_tuple in enumerate(axes):
            for j, ax in enumerate(ax_tuple):
                x_train, x_test = normalize_train_test(X_red_train[index], X_red_test[index])
                plot = ax.scatter(x_train[:, 0], x_train[:, 1], alpha=0.5,
                                  c=one_hot_to_num(Y)*2-4, cmap=cmap)
                # KNN section
                clf, xx, yy, Z = train_KNN(x_train, Y_train, n_neighbors)
                # getting test accuracy
                pred = clf.predict(x_test)
                Accuracy = sum(1 for i in range(len(pred)) if pred[i] == Y_test[i])/len(pred) 
                markers = []
                for i in range(len(pred)):
                    if pred[i] == Y_test[i]:
                        markers.append('o')
                    else:
                        markers.append('x')
                # plotting regions
                ax.pcolormesh(xx, yy, Z, cmap=cmap, alpha=0.1)
                # train vs test
                for i in range(len(x_test)):
                    ax.plot(x_test[i, 0], x_test[i, 1], marker=markers[i],
                            c='k', markersize=10, fillstyle='none')
                index += 1
        
        ticks=Oxys
        title = 'Oxidation'
        loc = [-1.33, 0, 1.33]
        label = title
        
    elif PROPERTY == 'category':
        
        title = "Category"
        label = title
            
        title = 'Aromatic'
            
        t1a = mpatches.Patch(color=Colors[0], label='Type 1')
        t1b = mpatches.Patch(color=Colors[1], label='Type 1')
        t2a = mpatches.Patch(color=Colors[2], label='Type 2')
        t2b = mpatches.Patch(color=Colors[3], label='Type 2')
        t3a = mpatches.Patch(color=Colors[4], label='Type 3')
        t3b = mpatches.Patch(color=Colors[5], label='Type 3')
        t4a = mpatches.Patch(color=Colors[6], label='Type 4')
        t4b = mpatches.Patch(color=Colors[7], label='Type 4')
        t5a = mpatches.Patch(color=Colors[8], label='Type 5')
        t5b = mpatches.Patch(color=Colors[9], label='Type 5')
        space = mpatches.Patch(color='w', label='')
        aro = mpatches.Patch(color='w', label='Aromatic')
        ali = mpatches.Patch(color='w', label='Aliphatic')
            
        handles = [ali, t1a,t2a, t3a, t4a, t5a, space, aro, t1b, t2b, t3b, t4b, t5b]
        leg1 = plt.legend(handles=handles, ncol=1, fontsize=20, handletextpad=0.001,
                              bbox_to_anchor=(1.03, 2.23), loc=2, borderaxespad=0.)
        plt.gca().add_artist(leg1)
        i = 0
        for patch in leg1.get_patches():
            patch.set_height(15)
            if i in [0, 7]:
                patch.set_width(0.1)
            else:
                patch.set_width(30)
            i += 1
            
        cbar_vis = False
        
        cmap = ListedColormap(Colors)
        
        index = 0
        for i, ax_tuple in enumerate(axes):
            for j, ax in enumerate(ax_tuple):
                x_train, x_test = normalize_train_test(X_red_train[index], X_red_test[index])
                plot = ax.scatter(x_train[:, 0], x_train[:, 1], alpha=0.5,
                                  c=one_hot_to_num(Y), cmap=cmap)
                # KNN section
                clf, xx, yy, Z = train_KNN(x_train, Y_train, n_neighbors)
                # getting test accuracy
                pred = clf.predict(x_test)
                Accuracy = sum(1 for i in range(len(pred)) if pred[i] == Y_test[i])/len(pred) 
                markers = []
                for i in range(len(pred)):
                    if pred[i] == Y_test[i]:
                        markers.append('o')
                    else:
                        markers.append('x')
                # plotting regions
                ax.pcolormesh(xx, yy, Z, cmap=cmap, alpha=0.1)
                # train vs test
                for i in range(len(x_test)):
                    ax.plot(x_test[i, 0], x_test[i, 1], marker=markers[i],
                            c='k', markersize=10, fillstyle='none')
                index += 1
        
    elif PROPERTY == 'Type':
        Colors = COLORS
        cmap = ListedColormap(Colors)
        
        index = 0
        for i, ax_tuple in enumerate(axes):
            for j, ax in enumerate(ax_tuple):
                x_train, x_test = normalize_train_test(X_red_train[index], X_red_test[index])
                plot = ax.scatter(x_train[:, 0], x_train[:, 1], alpha=0.5,
                                  c=one_hot_to_num(Y)*2-4, cmap=cmap)
                # KNN section
                clf, xx, yy, Z = train_KNN(x_train, Y_train, n_neighbors)
                # getting test accuracy
                pred = clf.predict(x_test)
                Accuracy = sum(1 for i in range(len(pred)) if pred[i] == Y_test[i])/len(pred) 
                markers = []
                for i in range(len(pred)):
                    if pred[i] == Y_test[i]:
                        markers.append('o')
                    else:
                        markers.append('x')
                # plotting regions
                ax.pcolormesh(xx, yy, Z, cmap=cmap, alpha=0.1)
                # train vs test
                for i in range(len(x_test)):
                    ax.plot(x_test[i, 0], x_test[i, 1], marker=markers[i],
                            c='k', markersize=10, fillstyle='none')
                index += 1
        
        ticks=Types
        title = "Type"
        loc = [-1.2, 0.4, 2., 3.6, 5.2]
        label = title
    
    # x and o legend
    black_o = mlines.Line2D([], [], c='k', marker='o', linestyle='None', fillstyle='none',
                              markersize=10, label='Correct')
    black_x = mlines.Line2D([], [], c='k', marker='x', linestyle='None', fillstyle='none',
                              markersize=10, label='Incorrect')
    
    leg2 = plt.legend(handles=[black_o, black_x], ncol=1, fontsize=22, handletextpad=0.001,
                              bbox_to_anchor=(0.825, 2.96), loc=2, borderaxespad=0.)
    plt.gca().add_artist(leg2)
    
    
    index = 0
    for i, ax_tuple in enumerate(axes):
        for j, ax in enumerate(ax_tuple):
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            t = ax.text(
                0.1, 0.85, methods[index], size=fontsize, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", lw=2, alpha=0.8,
                          facecolor="w", edgecolor="none"))
            index += 1

    plt.subplots_adjust(hspace=0.0, wspace=0.)
    
    if cbar_vis:
        cbaxes = fig.add_axes([0.91, 0.2, 0.033, 0.6])
        cbar = plt.colorbar(plot, cax=cbaxes, ticks=ticks)
        cbar.set_label(label, fontsize=26)
        cbar.set_ticks(loc)
        cbar.ax.tick_params(labelsize=26)
        cbar.set_ticklabels(ticks)
        cbar.set_alpha(1.)
        cbar.draw_all()
    
    plt.suptitle(f'\n\n{mode}', fontsize=30)
    
    plt.show()


def KNN_tsne(X_red, train_labels, method, TEST_SIZE, n_neighbors = 40, scheme=2):

    D_train = X_red[:-TEST_SIZE,:]
    Y_train = one_hot_to_num(train_labels)
    
    D_test = X_red[-TEST_SIZE:,:]
        
    if scheme == 1:
        Y_test = TEST_OXY
        title = 'Oxidation'
    elif scheme == 2:
        Y_test = TEST_TYPE
        title= 'Type'
    else:
        Y_test = TEST_CAT
        title = 'Aromaticity'
    
    # normalize
    Std = np.std(D_train, axis=0)
    Mean = np.mean(D_train, axis=0)
    D_train -= Mean
    D_train /= Std
    D_test -= Mean
    D_test /= Std   
    
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(D_train, Y_train)
    
    x_min, x_max = np.min(D_train[:,0]), np.max(D_train[:,0])
    y_min, y_max = np.min(D_train[:,1]), np.max(D_train[:,1])
    
    buffer = np.abs(x_max - x_min)*0.05
    h = (np.abs(x_max - x_min) + 2*buffer) / 100
    
    xx, yy = np.meshgrid(np.arange(x_min-buffer, x_max+buffer, h), np.arange(y_min-buffer, y_max+buffer, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # getting test accuracy
    pred = clf.predict(D_test)
    Accuracy = sum(1 for i in range(len(pred)) if pred[i] == Y_test[i])/len(pred) 

    print(f"KNN on {method}: {title} (Accuracy: {Accuracy:.2f})")
    return Accuracy


def KNN_2D(dim_reducer, train_labels, method, X, X_test, n_neighbors=40, scheme=2, val=False):
    # Thank you to:
    # https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
    # sphx-glr-auto-examples-neighbors-plot-classification-py

    if val:
        X_train, Y_train = shuffle_xy(X, one_hot_to_num(train_labels))
        train_size = 657
        X_test = X_train[train_size:].copy()
        Y_test = Y_train[train_size:].copy()
        X_train = X_train[:train_size].copy()
        Y_train = Y_train[:train_size].copy()
        
        if method is 'VAE':
            D_train = dim_reducer.predict(X_train, batch_size=20)[0]
            D_test = dim_reducer.predict(X_test, batch_size=20)[0]
        else:
            dim_reducer = dim_reducer.fit(X_train)
            D_train = dim_reducer.transform(X_train)
            D_test = dim_reducer.transform(X_test)
    else:
        X_train, Y_train = shuffle_xy(X, one_hot_to_num(train_labels))
        
        if method is 'VAE':
            D_train = dim_reducer.predict(X_train, batch_size=20)[0]
            D_test = dim_reducer.predict(X_test, batch_size=5)[0]
        else:
            dim_reducer = dim_reducer.fit(X_train)
            D_train = dim_reducer.transform(X_train)
            D_test = dim_reducer.transform(X_test)
            
    if scheme == 1:
        Y_test = TEST_OXY
        title = 'Oxidation'
    elif scheme == 2:
        Y_test = TEST_TYPE
        title= 'Type'
    else:
        Y_test = TEST_CAT
        title = 'Aromaticity'
    
    # normalize:
    Std = np.std(D_train, axis=0)
    Mean = np.mean(D_train, axis=0)
    D_train -= Mean
    D_train /= Std
    D_test -= Mean
    D_test /= Std   
    
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(D_train, Y_train)
    
    x_min, x_max = np.min(D_train[:,0]), np.max(D_train[:,0])
    y_min, y_max = np.min(D_train[:,1]), np.max(D_train[:,1])
    
    buffer = np.abs(x_max - x_min)*0.05
    h = (np.abs(x_max - x_min) + 2*buffer) / 100
    
    xx, yy = np.meshgrid(np.arange(x_min-buffer, x_max+buffer, h), np.arange(y_min-buffer, y_max+buffer, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # getting test accuracy
    pred = clf.predict(D_test)
    Accuracy = sum(1 for i in range(len(pred)) if pred[i] == Y_test[i])/len(pred)

    print(f"KNN on {method}: {title} (Accuracy: {Accuracy:.2f})")
    return Accuracy


def plot(Data, PROPERTY, x_predict, y_predict, encoder, X, test_index=None, mode='VtC-XES',
         extra_name=None, a=.3, s=5, black=False):
     
    z_mean = encoder.predict(x_predict, batch_size=1)[0]
    
    fontsize=28
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    
    cbar_vis = True
    
    if PROPERTY == 'oxy':
        Colors = [COLORS[0], COLORS[2], COLORS[4]]
        cmap = ListedColormap(Colors)
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=one_hot_to_num(y_predict)*2-4, cmap=cmap)
        ticks = Oxys
        label = 'Oxidation'
        loc = [-1.33, 0, 1.33]
        
    elif PROPERTY == 'category':
        Colors = list(COLORMAP(np.arange(1,11)/9))
        label = "Category"
            
        for i in range(10):
            Colors[i] = (128/255,128/255,128/255,0.2)
            
        # type 1
        Colors[0] = COLORS[0].copy()
        Colors[1] = '#9F5F80'
        # type 2
        Colors[2] = '#03506F'
        Colors[3] = COLORS[1].copy() + (60/255,75/255,75/255,0.)
        # type 3
        Colors[4] = '#DB6400'
        Colors[5] = '#ffba93'
        # type 4
        Colors[6] = '#2b3016'
        Colors[7] = COLORS[3].copy()
        # type 5
        Colors[8] = '#ac3501'
        Colors[9] = COLORS[4]
            
        t1a = mpatches.Patch(color=Colors[0], label='Type 1')
        t1b = mpatches.Patch(color=Colors[1], label='Type 1')
        t2a = mpatches.Patch(color=Colors[2], label='Type 2')
        t2b = mpatches.Patch(color=Colors[3], label='Type 2')
        t3a = mpatches.Patch(color=Colors[4], label='Type 3')
        t3b = mpatches.Patch(color=Colors[5], label='Type 3')
        t4a = mpatches.Patch(color=Colors[6], label='Type 4')
        t4b = mpatches.Patch(color=Colors[7], label='Type 4')
        t5a = mpatches.Patch(color=Colors[8], label='Type 5')
        t5b = mpatches.Patch(color=Colors[9], label='Type 5')
        space = mpatches.Patch(color='w', label='')
        aro = mpatches.Patch(color='w', label='Aromatic')
        ali = mpatches.Patch(color='w', label='Aliphatic')
          
            
        fig.subplots_adjust(right=0.75)
        handles = [ali, t1a,t2a, t3a, t4a, t5a, space, aro, t1b, t2b, t3b, t4b, t5b]
        leg1 = plt.legend(handles=handles, ncol=1, fontsize=20, handletextpad=0.001,
                              bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.gca().add_artist(leg1)
        i = 0
        for patch in leg1.get_patches():
            patch.set_height(15)
            if i in [0, 7]:
                patch.set_width(0.1)
            else:
                patch.set_width(30)
            i += 1
        
        ax.annotate(f'{mode}:\nVAE',
            xy=(0.03,0.85), xycoords='axes fraction',
            textcoords='offset points',
            size=fontsize)
        
        cbar_vis = False
        
        # actually plotting
        cmap = ListedColormap(Colors)
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=one_hot_to_num(y_predict), cmap=cmap)

        
    elif PROPERTY == 'Type':
        Colors = COLORS
        cmap = ListedColormap(Colors)
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=one_hot_to_num(y_predict), cmap=cmap)
        ticks = Types
        label = 'Type'
        loc = [1.4, 2.2, 3, 3.8, 4.6]
        if test_index is not None:
            ztest = encoder.predict(TEST_XES)[0]
            if test_index is 'all':
                plt.scatter(ztest[:, 0], ztest[:, 1], c='r', s=35)
            else:
                plt.plot(ztest[test_index, 0], ztest[test_index, 1], 'r.', markersize=15)
        
    else:
        
        z_meanX = encoder.predict(X, batch_size=batch_size)[0]
        Y = one_hot_to_num(get_Property(Data, 'oxy'))
        
        Colors = [COLORS[0], COLORS[2], COLORS[4]]
        cmap = ListedColormap(Colors)
        plt.scatter(z_meanX[:, 0], z_meanX[:, 1], c=Y*2-4, cmap=cmap)
        if z_mean.shape[0] > 5:
            alpha, color, ms = a, 'k', s
            plt.plot(z_mean[:, 0], z_mean[:, 1], '.-', c=color, alpha=alpha, markersize=ms)
        else:
            alpha, ms = a, s
            for i,pt in enumerate(z_mean):
                if i in [2,3]:
                    c='k'
                else:
                    c='#eb5600'
                plt.plot(z_mean[i, 0], z_mean[i, 1], 'o', c=c, alpha=1, markersize=ms+5, fillstyle='none')
                plt.plot(z_mean[i, 0], z_mean[i, 1], 'o', c=c, alpha=1, markersize=ms+3, fillstyle='none')
                plt.plot(z_mean[i, 0], z_mean[i, 1], 'o', c=c, alpha=1, markersize=ms+1, fillstyle='none')
            extra_name = None
        label = 'Oxidation'
        ticks = Oxys
        loc = [-1.33, 0, 1.33]
        if extra_name is not None:
            x = np.array([get_x(extra_name)])
            extra_name = extra_name.lower().replace('_' , ' ')
            z_loc = encoder.predict(x, batch_size=1)[0]
            plt.plot(z_loc[:, 0], z_loc[:, 1], '.', c='r', alpha=1., markersize=15, label=extra_name)
            plt.legend(fontsize=20)
    
    if black:
        z_meanX = encoder.predict(X, batch_size=batch_size)[0]
        Y = one_hot_to_num(get_Property(Data, 'oxy'))
        plt.scatter(z_meanX[:, 0], z_meanX[:, 1], c='k')
        cbar_vis=False
    
    if cbar_vis:
        cbaxes = fig.add_axes([0.92, 0.15, 0.033, 0.7])
        cbar = plt.colorbar(cax=cbaxes, ticks=ticks)
        cbar.set_label(label, fontsize=26)
        cbar.set_ticks(loc)
        cbar.ax.tick_params(labelsize=26)
        cbar.set_ticklabels(ticks)
        if extra_name is None:
            legend = ax.legend(['VAE'], markerscale=0.01, handlelength=0, handletextpad=0,
                               fancybox=True, fontsize=fontsize+12)
            for item in legend.legendHandles:
                item.set_visible(False)
    
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    plt.show()

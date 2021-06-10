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
    plt.plot([6], [6], c='w', alpha=1.0, label='\n      XES')
    
    plt.plot(xes_loss, '--', c=COLORS[3], linewidth=2, label='Training Loss')
    plt.plot(xes_val_loss, '-', c=COLORS[3], linewidth=4, label='Validation Loss')

    ax.set_ylim(bottom=120)
    
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SHOW PLOTS
#===================================================================================================

import sys
import struct
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib

def saveFigure(fig, fname):
 if fig:
     fig.savefig(fname, dpi="figure")
     print(f"{fname} has been updated")

#===================================================================================================
plt.rcParams["lines.markersize"] = 37
plt.rcParams["lines.linewidth"] = 4
plt.rcParams["font.size"] = 42
x_ticks_font_size = 42
y_ticks_font_size = 42
legend_font_size = 34
axis_font_size   = 42

#===================================================================================================
mark_mult = 15
markevery = [m * mark_mult for m in [2, 3, 2, 3, 2, 3, 2, 3, 2]]
marker = ["d","v","*","D","o","*", "x","^","*"]
#marker = [None, None, None, None, None, None, None, None, None]
color = ["#e41a1c", "#377eb8", "#4daf4a", "#35978f", "#ff7f00", "#ffff33", "#a65628", "#878787", "#999999"]
linestyle = ["solid", "dashed", "solid", "solid","solid","dashed", "solid", "dashed", "solid"]

#===================================================================================================
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
#===================================================================================================
func_value_fig = None
grad_norm_fig  = None
iterate_fig    = None
rounds_fig     = None
grad_norm_fig_vs_bits    = None
sent_bits_plot     = None
func_value_vs_time = None

#===================================================================================================
# CONFIG DISPLAYING STYLE
show_semilog_y = True

# COMMENT NOT NEEDED PLOT

#func_value_fig = plt.figure(figsize=(12, 9))
#iterate_fig    = plt.figure(figsize=(12, 9))
#rounds_fig     = plt.figure(figsize=(12, 9))
grad_norm_fig_vs_bits    = plt.figure(figsize=(12, 9))
grad_norm_fig            = plt.figure(figsize=(12, 9))
grad_norm_fig_vs_rounds  = plt.figure(figsize=(12, 9))

#sent_bits_plot     = plt.figure(figsize=(12, 9))
#func_value_vs_time = plt.figure(figsize=(12, 9))
#===================================================================================================
if func_value_fig:
    func_value_fig_ax = func_value_fig.add_subplot(1, 1, 1)

if grad_norm_fig:
    grad_norm_fig_ax = grad_norm_fig.add_subplot(1, 1, 1)

if iterate_fig:
    iterate_fig_ax = iterate_fig.add_subplot(1, 1, 1)

if rounds_fig:
    rounds_fig_ax = rounds_fig.add_subplot(1, 1, 1)

if grad_norm_fig_vs_bits:
    grad_norm_fig_ax_vs_bits = grad_norm_fig_vs_bits.add_subplot(1, 1, 1)

if grad_norm_fig_vs_rounds:
    grad_norm_fig_ax_vs_rounds = grad_norm_fig_vs_rounds.add_subplot(1, 1, 1)

if sent_bits_plot:
    sent_bits_plot_ax = sent_bits_plot.add_subplot(1, 1, 1)

if func_value_vs_time:
    func_value_vs_time_ax = func_value_vs_time.add_subplot(1,1,1)

#===================================================================================================
g = 0
#===================================================================================================
def getAsciiString(file):
    res = []

    while True:
        symbol = file.read(1)
        if symbol == b'\0':
            break
        res.append(symbol.decode(encoding='ascii'))

    resultString = ''.join(res)
    return resultString

def getInt32(file):
    (value, ) = struct.unpack("i", file.read(4))
    return value

def getUint32(file):
    (value, ) = struct.unpack("I", file.read(4))
    return value

def getInt64(file):
    (value, ) = struct.unpack("q", file.read(8))
    return value

def getUint64(file):
    (value, ) = struct.unpack("Q", file.read(8))
    return value

def getFP32(file):
    (value, ) = struct.unpack("f", file.read(4))
    return value

def getFP64(file):
    (value, ) = struct.unpack("d", file.read(8))
    return value
#=======================================================================================================
f_prime = 0
files = None

if "--first-file-is-optimal-solution" in sys.argv:
    # Get Optimal Solution File
    source = open(sys.argv[1], "rb")

    binFname         = getAsciiString(source)
    algoName         = getAsciiString(source)
    compressor = getAsciiString(source)

    totalSamplesPerClient = getInt32(source)
    totalSamples          = getInt32(source)
    roundsTotal           = getInt32(source)

    nClients          = getInt32(source)
    d                 = getInt32(source)
    k_for_fednl_c_abs = getInt32(source)

    k_for_fednl_c_mult_d = getFP64(source)
    global_lr            = getFP64(source)
    alpha_lr             = getFP64(source)
    lambda_regul         = getFP64(source)

    for r in range(roundsTotal-1):
        getInt32(source)
        getFP64(source)
        getFP64(source)
        getFP64(source)
        getFP64(source)
        getFP64(source)
        getFP64(source)

    # Read last entry from file with optimal solution
    getInt32(source) # rounds
    print("Optimal solution L2 norm gradient: ", getFP64(source))
    f_prime                    = getFP64(source)
    x_prime_norm               = getFP64(source)
    transfered_bytes_prime     = getFP64(source)
    compute_time_seconds_prime = getFP64(source)
    source.close()

    # Files to process
    files = sys.argv[2:]
else:
    files = sys.argv[1:]

#=========================================================================================================================
# PROCESS ALL FILES
#=========================================================================================================================
for fname in files:
    if fname.startswith("--"):
        continue

    #===================================================================================================
    # OBTAIN CURRENT FILE INFO
    #===================================================================================================
    source = open(fname, "rb")
    dataset    = getAsciiString(source)
    algorithm  = getAsciiString(source)
    compressor = getAsciiString(source)

    totalSamplesPerClient = getInt32(source)
    totalSamples  = getInt32(source)
    roundsTotal   = getInt32(source)
    nClients      = getInt32(source)
    d             = getInt32(source)

    print(f"Simulation {fname} solution rounds: ", roundsTotal)

    k_for_fednl_c_abs    = getInt32(source)
    k_for_fednl_c_mult_d = getFP64(source)

    global_lr     = getFP64(source)
    alpha_lr      = getFP64(source)
    lambda_regul  = getFP64(source)

    rounds    = []
    gradsNorm = []
    gradsNormSquare = []
    funcValue = []
    xiNorm    = []
    sendBitsTotal = []
    sendBitsIndicies = []
    eclapsedTime = []

    for r in range(roundsTotal):
        rounds.append(getInt32(source))
        gradNorm = getFP64(source)

        gradsNorm.append(gradNorm)
        gradsNormSquare.append(gradNorm**2)

        funcValue.append(getFP64(source) - f_prime)
        xiNorm.append(getFP64(source))

        receivedBytesFromClientsForScalarsAccum = getFP64(source) * 8 / nClients
        receivedBytesFromClientsForIndiciesAccum = getFP64(source) * 8 / nClients

        sendBitsTotal.append(receivedBytesFromClientsForScalarsAccum + receivedBytesFromClientsForIndiciesAccum)
        sendBitsIndicies.append(receivedBytesFromClientsForIndiciesAccum)

        eclapsedTime.append(getFP64(source))
    source.close()

    #=========================================================================================================================
    if algorithm == "gd" or algorithm == "GD":
        algo_name = f"{algorithm} ($\\gamma={global_lr:g}, \\lambda={lambda_regul:g}, d={d}$)"
    else:
        if compressor == "Identical" or compressor == "identical":
            algo_name = f"{algorithm} Identical"
        elif compressor == "randk" or compressor == "RandK":
            algo_name = f"{algorithm} RandK[k={k_for_fednl_c_mult_d:g}d]"
        elif compressor == "seqk" or compressor == "RandSeqK":
            algo_name = f"{algorithm} RandSeqK[k={k_for_fednl_c_mult_d:g}d]"
        elif compressor == "topk" or compressor == "TopK":
            algo_name = f"{algorithm} TopK[k={k_for_fednl_c_mult_d:g}d]"
        elif compressor == "toplek" or compressor == "TopLEK":
            algo_name = f"{algorithm} TopLEK[k={k_for_fednl_c_mult_d:g}d]"
        elif compressor == "natural" or compressor == "Natural":
            algo_name = f"{algorithm} Natural"

    gtrain = (g + 0) % len(color)
    #================================================================================================================================
    if func_value_fig:
        fig = plt.figure(func_value_fig.number)
        ax = func_value_fig_ax
        if show_semilog_y:
            ax.semilogy(sendBitsTotal, funcValue, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain], linestyle=linestyle[gtrain], label = algo_name)
        else:
            ax.plot(sendBitsTotal, funcValue, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain], linestyle=linestyle[gtrain], label=algo_name)
    
        ax.set_xlabel('#bits/n', fontdict = {'fontsize': axis_font_size})
        ax.set_ylabel('$f(x^t) - f(x^*)$', fontdict = {'fontsize': axis_font_size})
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.set_title(f"{dataset.upper()} $d={d},n={nClients},n_i={totalSamples//nClients},\\lambda={lambda_regul}$")
        ax.grid(True)
        ax.legend(loc='best', fontsize = legend_font_size)
    
        ax.set_xscale('log', basex=2)
        ax.set_yscale('log', basey=10)
        plt.xticks(fontsize=x_ticks_font_size)
        plt.yticks(fontsize=y_ticks_font_size)
        fig.tight_layout()
    #================================================================================================================================
    if grad_norm_fig:
        fig = plt.figure(grad_norm_fig.number)
        ax = grad_norm_fig_ax
        if show_semilog_y:
            ax.semilogy(eclapsedTime, gradsNorm, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                        linestyle=linestyle[gtrain], label=algo_name)
        else:
            ax.plot(eclapsedTime, gradsNorm, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                    linestyle=linestyle[gtrain], label=algo_name)
        ax.set_xlabel('Elapsed Time (seconds)', fontdict={'fontsize': axis_font_size})
        ax.set_ylabel('$\\|\\nabla f(x^t)\\|$', fontdict={'fontsize': axis_font_size})
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.set_title(f"{dataset.upper()} $d={d},n={nClients},n_i={totalSamples//nClients},\\lambda={lambda_regul}$")
        ax.grid(True)
        ax.legend(loc='best', fontsize=legend_font_size)
    
        #ax.set_xscale('log', basex=2)
        #ax.set_yscale('log', basey=10)
        plt.xticks(fontsize=x_ticks_font_size)
        plt.yticks(fontsize=y_ticks_font_size)
        fig.tight_layout()
    #=========================================================================================================================
    if grad_norm_fig_vs_bits:
        fig = plt.figure(grad_norm_fig_vs_bits.number)
        ax = grad_norm_fig_ax_vs_bits
        if show_semilog_y:
            ax.semilogy(sendBitsTotal, gradsNorm, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                        linestyle=linestyle[gtrain], label=algo_name)
        else:
            ax.plot(sendBitsTotal, gradsNorm, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                    linestyle=linestyle[gtrain], label=algo_name)
        ax.set_xlabel('#bits/n', fontdict = {'fontsize': axis_font_size})
        ax.set_ylabel('$\\|\\nabla f(x^t)\\|$', fontdict={'fontsize': axis_font_size})
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.set_title(f"{dataset.upper()} $d={d},n={nClients},n_i={totalSamples//nClients},\\lambda={lambda_regul}$")
        ax.grid(True)
        ax.legend(loc='best', fontsize=legend_font_size)
    
        #ax.set_xscale('log', basex=2)
        #ax.set_yscale('log', basey=10)
        plt.xticks(fontsize=x_ticks_font_size)
        plt.yticks(fontsize=y_ticks_font_size)
        fig.tight_layout()
    #=========================================================================================================================
    if grad_norm_fig_vs_rounds:
        fig = plt.figure(grad_norm_fig_vs_rounds.number)
        ax = grad_norm_fig_ax_vs_rounds
        if show_semilog_y:
            ax.semilogy(rounds, gradsNorm, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                        linestyle=linestyle[gtrain], label=algo_name)
        else:
            ax.plot(sendBitsTotal, gradsNorm, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                    linestyle=linestyle[gtrain], label=algo_name)
        ax.set_xlabel('Rounds', fontdict = {'fontsize': axis_font_size})
        ax.set_ylabel('$\\|\\nabla f(x^t)\\|$', fontdict={'fontsize': axis_font_size})
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.set_title(f"{dataset.upper()} $d={d},n={nClients},n_i={totalSamples//nClients},\\lambda={lambda_regul}$")
        ax.grid(True)
        ax.legend(loc='best', fontsize=legend_font_size)
    
        #ax.set_xscale('log', basex=2)
        #ax.set_yscale('log', basey=10)
        plt.xticks(fontsize=x_ticks_font_size)
        plt.yticks(fontsize=y_ticks_font_size)
        fig.tight_layout()
    #=========================================================================================================================

    if iterate_fig:
        fig = plt.figure(iterate_fig.number)
        ax = iterate_fig_ax
    
        if show_semilog_y:
            ax.semilogy(rounds, xiNorm, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                        linestyle=linestyle[gtrain], label=algo_name)
        else:
            ax.plot(rounds, xiNorm, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                    linestyle=linestyle[gtrain], label=algo_name)
        ax.set_xlabel('Rounds', fontdict={'fontsize': axis_font_size})
        ax.set_ylabel('$\\|x^t\\|$', fontdict={'fontsize': axis_font_size})
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.set_title(f"{dataset.upper()} $d={d},n={nClients},n_i={totalSamples//nClients},\\lambda={lambda_regul}$")
        ax.grid(True)
        ax.legend(loc='best', fontsize=legend_font_size)
        plt.xticks(fontsize = axis_font_size)
        plt.yticks(fontsize = axis_font_size)
        fig.tight_layout()
    #=========================================================================================================================
    if rounds_fig:
        fig = plt.figure(rounds_fig.number)
        ax = rounds_fig_ax
    
        if show_semilog_y:
            ax.semilogy(rounds, eclapsedTime, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                        linestyle=linestyle[gtrain], label=algo_name)
        else:
            ax.plot(rounds, eclapsedTime, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                    linestyle=linestyle[gtrain], label=algo_name)
        ax.set_xlabel('Rounds', fontdict={'fontsize': axis_font_size})
        ax.set_ylabel('Elapsed Time (seconds)', fontdict={'fontsize': axis_font_size})
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.set_title(f"{dataset.upper()} $d={d},n={nClients},n_i={totalSamples//nClients},\\lambda={lambda_regul}$")
        ax.grid(True)
        ax.legend(loc='best', fontsize=legend_font_size)
        plt.xticks(fontsize=axis_font_size)
        plt.yticks(fontsize=axis_font_size)
        fig.tight_layout()
    #=========================================================================================================================
    if sent_bits_plot:
        fig = plt.figure(sent_bits_plot.number)
        ax = sent_bits_plot_ax
        show_semilog_for_it = False
        if show_semilog_for_it:
            ax.semilogy(rounds, sendBitsIndicies, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                      linestyle=linestyle[gtrain], label=algo_name)
        else:
            ax.plot(rounds, sendBitsIndicies, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain],
                    linestyle=linestyle[gtrain], label=algo_name)
        ax.set_xlabel('Rounds', fontdict={'fontsize': axis_font_size})
        ax.set_ylabel('Communicated Bits/Node [Indicies]', fontdict = {'fontsize': axis_font_size})
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.set_title(f"{dataset.upper()} $d={d},n={nClients},n_i={totalSamples//nClients},\\lambda={lambda_regul}$")
        ax.grid(True)
        ax.legend(loc='best', fontsize=legend_font_size)
        plt.xticks(fontsize=axis_font_size)
        plt.yticks(fontsize=axis_font_size)
        fig.tight_layout()
    #=========================================================================================================================
    if func_value_vs_time:
        fig = plt.figure(func_value_vs_time.number)
        ax = func_value_vs_time_ax
        if show_semilog_y:
            ax.semilogy(eclapsedTime, funcValue, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain], linestyle=linestyle[gtrain], label = algo_name)
        else:
            ax.plot(eclapsedTime, funcValue, color=color[gtrain], marker=marker[gtrain], markevery=markevery[gtrain], linestyle=linestyle[gtrain], label=algo_name)
    
        ax.set_xlabel('Elapsed Time (seconds)', fontdict={'fontsize': axis_font_size})
        ax.set_ylabel('$f(x^t) - f(x^*)$', fontdict = {'fontsize': axis_font_size})
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.set_title(f"{dataset.upper()} $d={d},n={nClients},n_i={totalSamples//nClients},\\lambda={lambda_regul}$")
        ax.grid(True)
        ax.legend(loc='best', fontsize = legend_font_size)

        #ax.set_xscale('log', basex=2)
        #ax.set_yscale('log', basey=10)

        plt.xticks(fontsize=x_ticks_font_size)
        plt.yticks(fontsize=y_ticks_font_size)
        fig.tight_layout()
    #================================================================================================================================
    g = (g + 1) % len(color)
    #=========================================================================================================================

if "--save-fig" in sys.argv:
    print("Save figures in files")
    saveFigure(func_value_fig, "figure_1.pdf")
    saveFigure(grad_norm_fig, "figure_2.pdf")
    saveFigure(iterate_fig, "figure_3.pdf")
    saveFigure(rounds_fig, "figure_4.pdf")
    saveFigure(grad_norm_fig_vs_bits, "figure_5.pdf")
    saveFigure(sent_bits_plot, "figure_6.pdf")
    saveFigure(func_value_vs_time, "figure_7.pdf")

if "--no-gui" in sys.argv:
    print("Plotter has been launched with the request to not use GUI")
else:
    # Show plots in GUI
    plt.show()

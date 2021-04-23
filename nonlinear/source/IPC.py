import numpy as np
import scipy
from scipy.special import legendre
from scipy import sparse
from scipy.sparse import lil_matrix
import itertools
import argparse
import time
from datetime import timedelta
import sys

import os
import pickle
from loginit import get_module_logger


class IPCParams:
    def __init__(self, max_delay, max_deg, max_num_var, max_window, thres, deg_delays,\
        chunk=1000, max_capa=1000000, broke_thres=1):
        self.max_delay = max_delay
        self.max_deg = max_deg
        self.max_num_var = max_num_var
        self.max_window = max_window
        self.thres = thres
        self.deg_delays = deg_delays
        self.chunk = chunk
        self.max_capa = max_capa
        self.broke_thres = broke_thres

class IPC:
    def __init__(self, ipcparams, log, savedir=None, label=''):
        self.max_delay = ipcparams.max_delay
        self.max_deg = ipcparams.max_deg
        self.max_num_var = ipcparams.max_num_var
        self.thres = ipcparams.thres
        self.chunk = ipcparams.chunk
        self.max_capa = ipcparams.max_capa
        self.broke_thres = ipcparams.broke_thres
        self.max_window = ipcparams.max_window
        self.deg_delays = ipcparams.deg_delays

        self.log = log
        self.label = label
        if savedir == None:
            tmpdir = os.path.join(os.path.dirname(__file__), 'results')
        else:
            tmpdir = savedir
        self.savedir = tmpdir
    
    def run(self, input_signals, output_signals):
        self.__loop_IPC(input_signals, output_signals)
    
    def __loop_IPC(self, input_signals, output_signals):
        # Prepare signals
        start_time = time.monotonic()
        fname = '{}_{}'.format(self.label, sys._getframe().f_code.co_name)

        self.log.info('{}: Prepare signals...'.format(fname))
        self.num_data = input_signals.shape[0]
        input_legendre_arr = []

        mdeg = min(self.max_deg, len(self.deg_delays) - 1)
        for d in range(mdeg + 1):
            input_legendre_arr.append( legendre(d)(input_signals.reshape(-1)) )
        input_legendre_arr = np.array(input_legendre_arr)
        self.log.debug('{}: Created legendre input {}'.format(fname, input_legendre_arr.shape))
        
        total_capa = 0
        self.ipc_by_deg = np.zeros((2, mdeg + 1))
        self.ipc_by_deg[0] = np.arange(0, mdeg + 1)
        self.ipc_rs = dict()
        ipc_arr = dict()
        for stepbg in [0, 1]:
            for deg in range(stepbg, mdeg + 1, 2):
                ipc_arr[deg] = np.array([])
                capa_deg_ls = []
                local_max_delay = self.deg_delays[deg] - 1
                if local_max_delay < 0:
                    continue
                out = output_signals[local_max_delay:]
                avg = np.average(out, axis=0).reshape((-1, out.shape[1]))
                out -= avg
                N = out.shape[0]
                # calculate correlation
                inv_corr_mat_data = np.linalg.pinv((out.T@out)/float(N))

                if deg == 0:
                    target_signals = np.ones(self.num_data - local_max_delay).reshape(1,-1)
                    corr_mat_data_target = target_signals@out/float(N)
                    tmp = corr_mat_data_target[0]@inv_corr_mat_data@corr_mat_data_target[0].T/1.0
                    capa_deg_ls.append(tmp)
                else:
                    for nvar in range(1, min(self.max_num_var, deg) + 1):
                        # broke_wd = 0
                        # window = (max_delay - min_delay + 1) from min_delay to local_delay-1
                        for window in range(nvar, min(self.max_window + 1, local_max_delay + 2)):
                            # if broke_wd >= self.broke_thres:
                            #     self.log.debug('{}: Break in wd ={}; total={},wd={},nvar={},deg={}'.format(\
                            #             fname, broke_wd, total_capa, window, nvar, deg))
                            #     break

                            itdefault = [0, window-1]
                            if nvar > 2:
                                it_dis = [ list(it) + itdefault for it in itertools.combinations(list(range(1, window-1)), nvar - 2) ]
                            elif nvar == 2:
                                it_dis = [itdefault]
                            else:
                                it_dis = [[0]]
                            if nvar == 1 and window > 1:
                                break
                            nrm = deg - nvar
                            #broke_pos = 0
                            for it1 in it_dis:
                                # if broke_pos >= self.broke_thres:
                                #     self.log.debug('{}: Break in pos: break (wd,pos)=({},{}); total={},it1={},wd={},nvar={},deg={}'.format(\
                                #         fname, broke_wd, broke_pos, total_capa, it1, window, nvar, deg))
                                #     broke_wd += 1
                                #     break

                                if nrm > 0:
                                    it_sub = itertools.combinations_with_replacement(it1, nrm)
                                    its = [it1 + list(it2) for it2 in it_sub]
                                else:
                                    its = [it1]

                                for it in its:
                                    target_signals = []
                                    for sdelay in range(local_max_delay - window + 2):
                                        lg_val = np.ones(self.num_data - local_max_delay)
                                        for a in set(it):
                                            i = a + sdelay
                                            # Calculate IPC term
                                            bg = int(local_max_delay - i) 
                                            ed = int(self.num_data - i)
                                            tidx = it.count(a)
                                            lg_val *= input_legendre_arr[ tidx, bg:ed ]
                                        target_signals.append(lg_val)
                                        
                                    target_signals = np.array(target_signals)
                                    variance = np.var(target_signals, axis=1)
                                    corr_mat_data_target = target_signals@out/float(N)
                                    
                                    # broke_delay = 0
                                    for k in range(corr_mat_data_target.shape[0]):
                                        scapa = corr_mat_data_target[k]@inv_corr_mat_data@corr_mat_data_target[k].T/variance[k]
                                        # if scapa < self.thres:
                                        #     #break
                                        #     continue
                                            # broke_delay += 1
                                            # if broke_delay >= self.broke_thres:
                                            #     #self.log.debug('{}: Break in delay scapa={} (< {}); break (wd,pos,delay)=({},{},{}); total={},k={},it={},wd={},nvar={},deg={}'.format(\
                                            #     #    fname, scapa, self.thres, broke_wd, broke_pos, broke_delay, total_capa, k, it, window, nvar, deg))
                                            #     # broke_pos += 1
                                            #     break
                                        # total_capa += scapa
                                        # if total_capa > self.max_capa:
                                        #     self.log.info('{}: Break total_capa={} (over {}); k={},it={},wd={},nvar={},deg={}'.format(\
                                        #         fname, total_capa, self.max_capa, k, it, window, nvar, deg))
                                        #     broke_pos = self.broke_thres + 1
                                        #     broke_wd = self.broke_thres + 1
                                        #     break
                                        capa_deg_ls.append(scapa)
                
                capa_deg_ls = np.array(capa_deg_ls)
                tmp = np.sum(capa_deg_ls[capa_deg_ls >= self.thres])
                
                self.ipc_by_deg[1][deg] = tmp
                ipc_arr[deg] = capa_deg_ls
                self.log.info('Deg = {}, capa={}'.format(deg, tmp))
                                
        self.ipc_rs['ipc_arr'] = ipc_arr                        
        # Total capacity
        self.total_capacity = np.sum(self.ipc_by_deg[1])
        self.log.info('Total capacity = {}'.format(self.total_capacity))


    def write_results(self, posfix='capa'):
        start_time = time.monotonic()
        fname = '{}_{}'.format(self.label, sys._getframe().f_code.co_name)

        degdir = os.path.join(self.savedir,'degrs')
        if os.path.isdir(degdir) == False:
            os.mkdir(degdir)
        np.savetxt(os.path.join(degdir, 'degree_{}.txt'.format(posfix)), self.ipc_by_deg.T)
        self.log.info('{}: Finish write to degree.'.format(fname))

        self.ipc_rs['max_deg'] = self.max_deg
        self.ipc_rs['max_num_var'] = self.max_num_var
        self.ipc_rs['max_window'] = self.max_window
        self.ipc_rs['deg_delays'] = self.deg_delays
        self.ipc_rs['num_data'] = self.num_data
        ipcdir = os.path.join(self.savedir,'ipc')
        if os.path.isdir(ipcdir) == False:
            os.mkdir(ipcdir)
        with open(os.path.join(ipcdir, 'ipc_{}.pickle'.format(posfix)), "wb") as wfile:
            # Pickle the "data" dictionary using the highest protocol available.
            pickle.dump(self.ipc_rs, wfile, pickle.HIGHEST_PROTOCOL)
        
        self.log.info('{}: Finish dump to pickle.'.format(fname))

        end_time = time.monotonic()
        self.log.info('{}: Finish write to results. Executed time {}'.format(fname, timedelta(seconds=end_time - start_time)))
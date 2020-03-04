import sys
import numpy as np

def make_data_for_narma(length, orders):
    x = np.random.rand(length) * 0.2
    N = len(orders)
    Y = np.zeros((length, N))
    for j in range(N):
        order = orders[j]
        y = np.zeros(length)
        if order == 2:
            for i in range(length):
                y[i] = 0.4 * y[i-1] + 0.4 * y[i-1]*y[i-2] + 0.6 * (x[i]**3) + 0.1
        else:
            for i in range(length):
                if i < order:
                    y[i] = 0.3 * y[i - 1] + 0.05 * y[i - 1] * np.sum(np.hstack((y[i - order:], y[:i]))) + \
                        1.5 * x[i - order + 1] * x[i] + 0.1
                else:
                    y[i] = 0.3 * y[i - 1] + 0.05 * y[i - 1] * np.sum(np.hstack((y[i - order:i]))) + \
                        1.5 * x[i - order + 1] * x[i] + 0.1
        Y[:,j] = y
    return x, Y

def generate_data(sequence_count, sequence_length, delay):
    input_sequence_list = []
    for sequence_index in range(sequence_count):
        r = sequence_index%6
        if r==0:
            input_sequence = sinwave(3.+np.random.rand()*3, np.random.rand()*2*np.pi, sequence_length)
        elif r==1:
            input_sequence = sqwave(3.+np.random.rand()*3, np.random.rand()*2*np.pi, sequence_length)
        elif r==2:
            input_sequence = triwave(3.+np.random.rand()*3, np.random.rand()*2*np.pi, sequence_length)
        elif r==3:
            input_sequence = sawwave(3.+np.random.rand()*3, np.random.rand()*2*np.pi, sequence_length)
        elif r==4:
            input_sequence = moving_noise(np.random.rand(), np.random.rand(), np.random.rand(), sequence_length)
        elif r==5:
            input_sequence = fm_sinwave(3.+np.random.rand()*3, np.random.rand()*2*np.pi, np.random.rand(), np.random.rand()*5, sequence_length)
        else:
            assert(0<=r and r<6)
            exit(0)
        input_sequence_list.append(input_sequence)

    output_sequence_list = []
    for input_sequence in input_sequence_list:
        output_sequence = generate_echo_sequence(input_sequence, delay)
        output_sequence_list.append(output_sequence)

    input_sequence_list = np.array(input_sequence_list)
    output_sequence_list = np.array(output_sequence_list)

    assert(input_sequence_list.shape == (sequence_count, sequence_length))
    assert(output_sequence_list.shape == (sequence_count, sequence_length))

    return input_sequence_list, output_sequence_list

    
def sinwave(freq, phase, sampling_freq):
    """sinwave
    """
    phase_series = np.linspace(0,1,sampling_freq) * freq + phase/2/np.pi
    sequence = np.sin( phase_series*2*np.pi )
    return sequence

def sqwave(freq,phase,sampling_freq):
    """square wave
    """
    phase_series = np.linspace(0,1,sampling_freq) * freq + phase/2/np.pi
    sequence = (np.mod(phase_series,1)<0.5).astype(np.float64)*2-1
    return sequence

def sawwave(freq, phase, sampling_freq):
    """saw wave
    """
    phase_series = np.linspace(0,1,sampling_freq) * freq + phase/2/np.pi
    sequence = np.mod((phase_series*2+1.),2.)-1.
    return sequence

def triwave(freq,phase,sampling_freq):
    """triangle wave
    """
    phase_series = np.linspace(0,1,sampling_freq) * freq + phase/2/np.pi
    sequence = -np.abs(np.mod(phase_series+0.25,1.)-0.5)*4+1
    return sequence

def moving_noise(force,damp,cov,sampling_freq,seed=-1):
    """continuously moving noise signal
    """
    if seed>0:
        np.random.seed(seed)
    x = 0.
    v = np.random.rand()
    sequence = [x]
    for _ in range(sampling_freq-1):
        v += np.random.normal()*force - cov*v - damp * x
        x += v
        sequence.append(sequence[-1]+v)
    # 後の手続きのため、[-1,1]の範囲に正規化する
    sequence -= (np.max(sequence) + np.min(sequence))/2
    sequence/= np.max(np.abs(sequence))
    return sequence

def fm_sinwave(freq, phase, fm_amp, fm_freq, sampling_freq):
    """frequency modulated sinwave
    """
    time_series = np.linspace(0,1,sampling_freq)
    phase_series = time_series * freq + phase/2/np.pi + fm_amp * np.sin(fm_freq*time_series*np.pi*2)
    sequence = np.sin( phase_series*2*np.pi )
    return sequence

def generate_echo_sequence(sequence, delay):
    sequence = np.roll(sequence,delay)

    # 存在しない信号については0で埋める。
    if delay > 0:
        sequence[:delay]=0
    elif delay<0:
        sequence[delay:]=0
    return sequence


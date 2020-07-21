import essentia
import essentia.standard as es
import numpy as np

def spectrogram(audio, ws=2048, hs=256, scale='dBV'):
    '''
    Args:
        audio (array): audio signal (output from MonoLoader)
        ws (int): window size
        hp (int): hop size
        scale (str): scale for the z axis: 'linear', 'dBV'
    Returns:
        x_axis (array): x axis as time in seconds
        y_axis (array): y axis as frequency
        z_axis (array): z axis according to the selected scale
    '''
    # minimum number possible
    eps = np.finfo(np.float).eps

    # instantiate Essentia functions
    w = es.Windowing(type='hann', zeroPadding=ws)
    spec = es.Spectrum()

    # empty list for spectra
    mX = []

    # iterate over frames
    for frame in es.FrameGenerator(audio, frameSize=ws, hopSize=hs):
        mXFrame = spec(w(frame))
        mX.append(mXFrame)

    # convert list to array and transpose
    mX = np.array(mX)
    mX = np.transpose(mX)

    # compute plot axes
    x_axis = np.arange(mX.shape[1]) * hs / 44100
    y_axis = np.arange(mX.shape[0]) * 44100 / (ws * 2)
    if scale == 'linear':
        z_axis = mX
    elif scale == 'dBV':
        z_axis = 20*np.log10(mX+eps)

    return x_axis, y_axis, z_axis

def get_f0(audio, minf0=20, maxf0=22050, cf=0.9, ws=2048, hs=256):
        '''
        Args:
            audio (array): audio signal (output from MonoLoader)
            minf0 (int): minimum allowed frequency
            maxf0 (int): maximun allowed frequency
            cf (float): confidence threshold (0 - 1)
            ws (int): window size
            hp (int): hop size

        Returns:
            f0 (array):
        '''
        # instantiate Essentia functions
        w = es.Windowing(type='hann', zeroPadding=ws)
        spec = es.Spectrum()
        yin = es.PitchYinFFT(minFrequency=minf0, maxFrequency=maxf0, frameSize=ws)

        # empty lists for f0 and confidence
        f0 = []
        conf = []

        # iterate over frames
        for frame in es.FrameGenerator(audio, frameSize=ws, hopSize=hs):
            p, pc = yin(spec(w(frame)))
            f0.append(p)
            conf.append(pc)

        # convert lists to np.arrays
        f0 = np.array(f0)
        conf = np.array(conf)

        # return f0 over given confidence
        f0[conf < cf] = 0
        return f0

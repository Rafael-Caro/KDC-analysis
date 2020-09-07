'''
Here I put together code from https://github.com/gopalkoduri/intonation, and
from https://github.com/gopalkoduri/pypeaks, since I couldn't install them
properly
'''


import pickle
import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import variation, skew, kurtosis
import pylab as p
from warnings import warn

################################################################################
# SLOPE                                                                        #
################################################################################

def slp_find_nearest_index(arr, value):
    """For a given value, the function finds the nearest value
    in the array and returns its index."""
    arr = np.array(arr)
    index = (np.abs(arr - value)).argmin()
    return index


#the following two functions are taken from:
#https://gist.github.com/1178136


def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise ValueError('Input vectors y_axis and x_axis must have same length')

    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def slp_peakdetect(y_axis, x_axis=None, lookahead=300, delta=0):
    """
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200)
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   # Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)


    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    #NOTE: commented this to use the function with log(histogram)
    #if not (np.isscalar(delta) and delta >= 0):
    if not (np.isscalar(delta)):
        raise ValueError("delta must be a positive number")

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]

    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass

    return [max_peaks, min_peaks]


def slp_peaks(x, y, lookahead=20, delta=0.00003):
    """
    A wrapper around peakdetect to pack the return values in a nicer format
    """
    _max, _min = slp_peakdetect(y, x, lookahead, delta)
    x_peaks = [p[0] for p in _max]
    y_peaks = [p[1] for p in _max]
    x_valleys = [p[0] for p in _min]
    y_valleys = [p[1] for p in _min]

    _peaks = [x_peaks, y_peaks]
    _valleys = [x_valleys, y_valleys]
    return {"peaks": _peaks, "valleys": _valleys}

################################################################################
# DATA                                                                         #
################################################################################

class Data:
    """
    The Data object for peak detection has methods necessary to handle the
    histogram/time-series like data.
    """
    def __init__(self, x, y, smoothness=7, default_smooth=True):
        """
        Initializes the data object for peak detection with x, y, smoothness
        parameter and peaks (empty initially). In a histogram, x refers to bin centers (not
        edges), and y refers to the corresponding count/frequency.
        The smoothness parameter refers to the standard deviation of the
        gaussian kernel used for smoothing.
        The peaks variable is a dictionary of the following form:
        {"peaks":[[peak positions], [peak amplitudes]],
        "valleys": [[valley positions], [valley amplitudes]]}
        """

        self.x = np.array(x)
        self.y_raw = np.array(y)
        self.y = np.array(y)
        self.smoothness = smoothness
        if default_smooth:
            self.smooth()
        self.peaks = {}

    def set_smoothness(self, smoothness):
        """
        This method (re)sets the smoothness parameter.
        """
        self.smoothness = smoothness
        self.smooth()

    def smooth(self):
        """
        Smooths the data using a gaussian kernel of the given standard deviation
        (as smoothness parameter)
        """
        self.y = gaussian_filter(self.y_raw, self.smoothness)

    def normalize(self):
        """
        Normalizes the given data such that the area under the histogram/curve
        comes to 1. Also re applies smoothing once done.
        """
        median_diff = np.median(np.diff(self.x))
        bin_edges = [self.x[0] - median_diff/2.0]
        bin_edges.extend(median_diff/2.0 + self.x)
        self.y_raw = self.y_raw/(self.y_raw.sum()*np.diff(bin_edges))
        self.smooth()

    def serialize(self, path):
        """
        Saves the raw (read unsmoothed) histogram data to the given path using
        pickle python module.
        """
        pickle.dump([self.x, self.y_raw], file(path, 'w'))

    def get_peaks(self, method="slope", peak_amp_thresh=0.00005,
                  valley_thresh=0.00003, intervals=None, lookahead=20,
                  avg_interval=100):
        """
        This function expects SMOOTHED histogram. If you run it on a raw histogram,
        there is a high chance that it returns no peaks.
        method can be interval/slope/hybrid.
            The interval-based method simply steps through the whole histogram
            and pick up the local maxima in each interval, from which irrelevant
            peaks are filtered out by looking at the proportion of points on
            either side of the detected peak in each interval, and by applying
            peal_amp_thresh and valley_thresh bounds.

            Slope approach uses, of course slope information, to find peaks,
            which are then filtered by applying peal_amp_thresh and
            valley_thresh bounds.

            Hybrid approach combines the peaks obtained using slope method and
            interval-based approach. It retains the peaks/valleys from slope method
            if there should be a peak around the same region from each of the methods.

        peak_amp_thresh is the minimum amplitude/height that a peak should have
        in a normalized smoothed histogram, to be qualified as a peak.
        valley_thresh is viceversa for valleys!
        If the method is interval/hybrid, then the intervals argument must be passed
        and it should be an instance of Intervals class.
        If the method is slope/hybrid, then the lookahead and avg_window
        arguments should be changed based on the application.
        They have some default values though.
        The method stores peaks in self.peaks in the following format:
        {"peaks":[[peak positions], [peak amplitudes]],
        "valleys": [[valley positions], [valley amplitudes]]}
        """

        peaks = {}
        slope_peaks = {}
        #Oh dear future me, please don't get confused with a lot of mess around
        # indices around here. All indices (eg: left_index etc) refer to indices
        # of x or y (of histogram).
        if method == "slope" or method == "hybrid":

            #step 1: get the peaks
            result = slp_peaks(self.x, self.y, lookahead=lookahead,
                                 delta=valley_thresh)

            #step 2: find left and right valley points for each peak
            peak_data = result["peaks"]
            valley_data = result["valleys"]

            for i in range(len(peak_data[0])):
                nearest_index = slp_find_nearest_index(valley_data[0],
                                                         peak_data[0][i])
                if valley_data[0][nearest_index] < peak_data[0][i]:
                    left_index = slp_find_nearest_index(
                        self.x, valley_data[0][nearest_index])
                    if len(valley_data[0][nearest_index + 1:]) == 0:
                        right_index = slp_find_nearest_index(
                            self.x, peak_data[0][i] + avg_interval / 2)
                    else:
                        offset = nearest_index + 1
                        nearest_index = offset + slp_find_nearest_index(
                            valley_data[0][offset:], peak_data[0][i])
                        right_index = slp_find_nearest_index(
                            self.x, valley_data[0][nearest_index])
                else:
                    right_index = slp_find_nearest_index(
                        self.x, valley_data[0][nearest_index])
                    if len(valley_data[0][:nearest_index]) == 0:
                        left_index = slp_find_nearest_index(
                            self.x, peak_data[0][i] - avg_interval / 2)
                    else:
                        nearest_index = slp_find_nearest_index(
                            valley_data[0][:nearest_index], peak_data[0][i])
                        left_index = slp_find_nearest_index(
                            self.x, valley_data[0][nearest_index])

                pos = slp_find_nearest_index(self.x, peak_data[0][i])
                slope_peaks[pos] = [peak_data[1][i], left_index, right_index]

        if method == "slope":
            peaks = slope_peaks

        interval_peaks = {}
        if method == "interval" or method == "hybrid":
            if intervals is None:
                raise ValueError('The interval argument is not passed.')
            #step 1: get the average size of the interval, first and last
            # probable centers of peaks
            avg_interval = np.average(intervals.intervals[1:] - intervals.intervals[:-1])
            value = (min(self.x) + 1.5 * avg_interval) / avg_interval * avg_interval
            first_center = intervals.nearest_interval(value)
            value = (max(self.x) - avg_interval) / avg_interval * avg_interval
            last_center = intervals.nearest_interval(value)
            if first_center < intervals.intervals[0]:
                first_center = intervals.intervals[0]
                warn("In the interval based approach, the first center was seen\
                    to be too low and is set to " + str(first_center))
            if last_center > intervals.intervals[-1]:
                last_center = intervals.intervals[-1]
                warn("In the interval based approach, the last center was seen\
                     to be too high and is set to " + str(last_center))

            #step 2: find the peak position, and set the left and right bounds
            # which are equivalent in sense to the valley points
            interval = first_center
            while interval <= last_center:
                prev_interval = intervals.prev_interval(interval)
                next_interval = intervals.next_interval(interval)
                left_index = slp_find_nearest_index(
                    self.x, (interval + prev_interval) / 2)
                right_index = slp_find_nearest_index(
                    self.x, (interval + next_interval) / 2)
                peak_pos = np.argmax(self.y[left_index:right_index])
                # add left_index to peak_pos to get the correct position in x/y
                peak_amp = self.y[left_index + peak_pos]
                interval_peaks[left_index + peak_pos] = [peak_amp, left_index,
                                                         right_index]
                interval = next_interval

        if method == "interval":
            peaks = interval_peaks

        # If its is a hybrid method merge the results. If we find same
        # peak position in both results, we prefer valleys of slope-based peaks
        if method == "hybrid":
            p1 = slope_peaks.keys()
            p2 = interval_peaks.keys()
            all_peaks = {}
            for p in p1:
                near_index = slp_find_nearest_index(p2, p)
                if abs(p - p2[near_index]) < avg_interval / 2:
                    p2.pop(near_index)
            for p in p1:
                all_peaks[p] = slope_peaks[p]
            for p in p2:
                all_peaks[p] = interval_peaks[p]
            peaks = all_peaks

        # Finally, filter the peaks and retain eligible peaks, also get
        # their valley points.

        # check 1: peak_amp_thresh
        # CHANGE: peaks.keys() --> list(peaks)
        for pos in list(peaks):
            # pos is an index in x/y. DOES NOT refer to a cent value.
            if peaks[pos][0] < peak_amp_thresh:
                peaks.pop(pos)

        # check 2, 3: valley_thresh, proportion of size of left and right lobes
        valleys = {}
        for pos in peaks.keys():
            # remember that peaks[pos][1] is left_index and
            # peaks[pos][2] is the right_index
            left_lobe = self.y[peaks[pos][1]:pos]
            right_lobe = self.y[pos:peaks[pos][2]]
            if len(left_lobe) == 0 or len(right_lobe) == 0:
                peaks.pop(pos)
                continue
            if len(left_lobe) / len(right_lobe) < 0.15 or len(right_lobe) / len(left_lobe) < 0.15:
                peaks.pop(pos)
                continue
            left_valley_pos = np.argmin(left_lobe)
            right_valley_pos = np.argmin(right_lobe)
            if (abs(left_lobe[left_valley_pos] - self.y[pos]) < valley_thresh and
                abs(right_lobe[right_valley_pos] - self.y[pos]) < valley_thresh):
                peaks.pop(pos)
            else:
                valleys[peaks[pos][1] + left_valley_pos] = left_lobe[left_valley_pos]
                valleys[pos + right_valley_pos] = right_lobe[right_valley_pos]

        if len(peaks) > 0:
            # CHANGE: peaks.values() --> list(peaks.values())
            peak_amps = np.array(list(peaks.values()))
            peak_amps = peak_amps[:, 0]
            # hello again future me, it is given that you'll pause here
            # wondering why the heck we index x with peaks.keys() and
            # valleys.keys(). Just recall that pos refers to indices and
            # not value corresponding to the histogram bin. If i is pos,
            # x[i] is the bin value. Tada!!
            # CHANGE: peaks.keys() --> list(peaks.keys())
            # CHANGE: valleys.keys() --> list(valleys.keys())
            # CHANGE: valleys.values() --> list(valleys.values())
            self.peaks = {'peaks': [self.x[list(peaks.keys())], peak_amps], 'valleys': [self.x[list(valleys.keys())], list(valleys.values())]}
        else:
            self.peaks = {'peaks': [[], []], 'valleys': [[], []]}

    def extend_peaks(self, prop_thresh=50):
        """Each peak in the peaks of the object is checked for its presence in
        other octaves. If it does not exist, it is created.

        prop_thresh is the cent range within which the peak in the other octave
        is expected to be present, i.e., only if there is a peak within this
        cent range in other octaves, then the peak is considered to be present
        in that octave.
        Note that this does not change the peaks of the object. It just returns
        the extended peaks.
        """

        # octave propagation of the reference peaks
        temp_peaks = [i + 1200 for i in self.peaks["peaks"][0]]
        temp_peaks.extend([i - 1200 for i in self.peaks["peaks"][0]])
        extended_peaks = []
        extended_peaks.extend(self.peaks["peaks"][0])
        for i in temp_peaks:
            # if a peak exists around, don't add this new one.
            nearest_ind = slp_find_nearest_index(self.peaks["peaks"][0], i)
            diff = abs(self.peaks["peaks"][0][nearest_ind] - i)
            diff = np.mod(diff, 1200)
            if diff > prop_thresh:
                extended_peaks.append(i)
        return extended_peaks

    # def plot(self, intervals=None, new_fig=True):
    #     """This function plots histogram together with its smoothed
    #     version and peak information if provided. Just intonation
    #     intervals are plotted for a reference."""
    #
    #     import pylab as p
    #
    #     if new_fig:
    #         p.figure()
    #
    #     #step 1: plot histogram
    #     p.plot(self.x, self.y, ls='-', c='b', lw='1.5')
    #
    #     #step 2: plot peaks
    #     first_peak = None
    #     last_peak = None
    #     if self.peaks:
    #         first_peak = min(self.peaks["peaks"][0])
    #         last_peak = max(self.peaks["peaks"][0])
    #         p.plot(self.peaks["peaks"][0], self.peaks["peaks"][1], 'rD', ms=10)
    #         p.plot(self.peaks["valleys"][0], self.peaks["valleys"][1], 'yD', ms=5)
    #
    #     #Intervals
    #     if intervals is not None:
    #         #spacing = 0.02*max(self.y)
    #         for interval in intervals:
    #             if first_peak is not None:
    #                 if interval <= first_peak or interval >= last_peak:
    #                     continue
    #             p.axvline(x=interval, ls='-.', c='g', lw='1.5')
    #             if interval-1200 >= min(self.x):
    #                 p.axvline(x=interval-1200, ls=':', c='b', lw='0.5')
    #             if interval+1200 <= max(self.x):
    #                 p.axvline(x=interval+1200, ls=':', c='b', lw='0.5')
    #             if interval+2400 <= max(self.x):
    #                 p.axvline(x=interval+2400, ls='-.', c='r', lw='0.5')
    #             #spacing *= -1
    #
    #     #p.title("Tonic-aligned complete-range pitch histogram")
    #     #p.xlabel("Pitch value (Cents)")
    #     #p.ylabel("Normalized frequency of occurence")
    #     p.show()

    def plot(self, intervals=None, new_fig=True, shahed=None,
             saveToFileName=None, title=None, xlabel=None, ylabel=None):
        """This function plots histogram together with its smoothed
        version and peak information if provided. Just intonation
        intervals are plotted for a reference."""

        import pylab as p

        if new_fig:
            p.figure()

        #step 1: plot histogram
        p.plot(self.x, self.y, ls='-', c='b', lw='1.5')

        #step 2: plot peaks
        first_peak = None
        last_peak = None
        if self.peaks:
            first_peak = min(self.peaks["peaks"][0])
            last_peak = max(self.peaks["peaks"][0])
            p.plot(self.peaks["peaks"][0], self.peaks["peaks"][1], 'r.', ms=10)
            if shahed != None:
                distance = 100
                shahed_index = 0
                for i in range(len(self.peaks["peaks"][0])):
                    current_peak = self.peaks["peaks"][0][i]
                    current_distance = abs(shahed - current_peak)
                    if current_distance < distance:
                        distance = current_distance
                        shahed_index = i
                p.plot(self.peaks["peaks"][0][shahed_index],
                       self.peaks["peaks"][1][shahed_index], 'rD', ms=10)
        p.title(title)
        p.ylabel(ylabel)
        p.xlabel(xlabel)

        #Intervals
        if intervals is not None:
            #spacing = 0.02*max(self.y)
            for interval in intervals:
                if first_peak is not None:
                    if interval <= first_peak or interval >= last_peak:
                        continue
                p.axvline(x=interval, ls='-.', c='g', lw='1.5')
                if interval-1200 >= min(self.x):
                    p.axvline(x=interval-1200, ls=':', c='b', lw='0.5')
                if interval+1200 <= max(self.x):
                    p.axvline(x=interval+1200, ls=':', c='b', lw='0.5')
                if interval+2400 <= max(self.x):
                    p.axvline(x=interval+2400, ls='-.', c='r', lw='0.5')
                #spacing *= -1

        #p.title("Tonic-aligned complete-range pitch histogram")
        #p.xlabel("Pitch value (Cents)")
        #p.ylabel("Normalized frequency of occurence")
        if saveToFileName == None:
            p.show()
        else:
            p.savefig(saveToFileName)
            p.show()

################################################################################
# PITCH                                                                        #
################################################################################

class Pitch:
    def __init__(self, timestamps, pitch):
        self.timestamps = timestamps
        self.pitch_raw = pitch
        self.pitch = pitch

    def reset(self):
        self.pitch = self.pitch_raw

    def discretize(self, intervals, slope_thresh=1500, cents_thresh=50):
        """
        This function takes the pitch data and returns it quantized to given
        set of intervals. All transactions must happen in cent scale.
        slope_thresh is the bound beyond which the pitch contour is said to transit
        from one svara to another. It is specified in cents/sec.
        cents_thresh is a limit within which two pitch values are considered the same.
        This is what pushes the quantization limit.
        The function returns quantized pitch data.
        """

        #eps = np.finfo(float).eps
        #pitch = median_filter(pitch, 7)+eps

        self.pitch = median_filter(self.pitch, 7)
        pitch_quantized = np.zeros(len(self.pitch))
        pitch_quantized[0] = find_nearest_index(intervals, self.pitch[0])
        pitch_quantized[-1] = find_nearest_index(intervals, self.pitch[-1])

        for i in range(1, len(self.pitch)-1):
            if self.pitch[i] == -10000:
                pitch_quantized[i] = -10000
                continue
            slope_back = abs((self.pitch[i] - self.pitch[i-1])/(self.timestamps[i] - self.timestamps[i-1]))
            slope_front = abs((self.pitch[i+1] - self.pitch[i])/(self.timestamps[i+1] - self.timestamps[i]))
            if slope_front < slope_thresh or slope_back < slope_thresh:
                ind = find_nearest_index(intervals, self.pitch[i])
                cents_diff = abs(self.pitch[i] - intervals[ind])
                if cents_diff <= cents_thresh:
                    pitch_quantized[i] = intervals[ind]
                else:
                    pitch_quantized[i] = -10000
            else:
                pitch_quantized[i] = -10000

        self.pitch = pitch_quantized

    def enforce_duration(self, duration_thresh):
        """
        This method takes a quantized pitch contour and filters out
        those time sections where the contour is not long enough, as specified
        by duration threshold (given in milliseconds).
        All transactions assume data in cent scale.
        """
        i = 1
        while i < len(self.pitch)-1:
            if self.pitch[i] == -10000:
                i += 1
                continue
            if self.pitch[i]-self.pitch[i-1] != 0 and self.pitch[i+1]-self.pitch[i] == 0:
                start = i
                while i < len(self.pitch) and self.pitch[i+1]-self.pitch[i] == 0:
                    i += 1
                if (self.timestamps[i]-self.timestamps[start])*1000 < duration_thresh:
                    self.pitch[start:i+1] = np.zeros(i+1-start)-10000
            else:
                self.pitch[i] = -10000
                i += 1

    def fit_lines(self, window=1500, break_thresh=1500):
        """
        Fits lines to pitch contours.
        :param window: size of each chunk to which linear equation is to be fit (in milliseconds).
        To keep it simple, hop is chosen to be one third of the window.
        :param break_thresh: If there is silence beyond this limit (in milliseconds),
        the contour will be broken there into two so that we don't fit a line over and
        including the silent region.
        """
        window /= 1000
        hop = window/3
        break_thresh /= 1000

        #cut the whole song into pieces if there are gaps more than break_thresh seconds
        i = 0
        break_indices = []
        count = 0
        while i < len(self.pitch):
            if self.pitch[i] == -10000:
                count = 1
                start_index = i
                while i < len(self.pitch) and self.pitch[i] == -10000:
                    count += 1
                    i += 1
                end_index = i-1
                if self.timestamps[end_index]-self.timestamps[start_index] >= break_thresh:
                    break_indices.append([start_index, end_index])
            i += 1
        break_indices = np.array(break_indices)

        #In creating the data blocks which are not silences, note that we
        # take complimentary break indices. i.e., if [[s1, e1], [s2, e2] ...]
        # is break_indices, we take e1-s2, e2-s3 chunks and build data blocks

        data_blocks = []
        if len(break_indices) == 0:
            t_pitch = self.pitch.reshape(len(self.pitch), 1)
            t_timestamps = self.timestamps.reshape(len(self.timestamps), 1)
            data_blocks = [np.append(t_timestamps, t_pitch, axis=1)]
        else:
            if break_indices[0, 0] != 0:
                t_pitch = self.pitch[:break_indices[0, 0]]
                t_pitch = t_pitch.reshape(len(t_pitch), 1)
                t_timestamps = self.timestamps[:break_indices[0, 0]]
                t_timestamps = t_timestamps.reshape(len(t_timestamps), 1)
                data_blocks.append(np.append(t_timestamps, t_pitch, axis=1))
            block_start = break_indices[0, 1]
            for i in range(1, len(break_indices)):
                block_end = break_indices[i, 0]
                t_pitch = self.pitch[block_start:block_end]
                t_pitch = t_pitch.reshape(len(t_pitch), 1)
                t_timestamps = self.timestamps[block_start:block_end]
                t_timestamps = t_timestamps.reshape(len(t_timestamps), 1)
                data_blocks.append(np.append(t_timestamps, t_pitch, axis=1))
                block_start = break_indices[i, 1]
            if block_start != len(self.pitch)-1:
                t_pitch = self.pitch[block_start:]
                t_pitch = t_pitch.reshape(len(t_pitch), 1)
                t_timestamps = self.timestamps[block_start:]
                t_timestamps = t_timestamps.reshape(len(t_timestamps), 1)
                data_blocks.append(np.append(t_timestamps, t_pitch, axis=1))

        label_start_offset = (window-hop)/2
        label_end_offset = label_start_offset+hop

        #dataNew = np.zeros_like(data)
        #dataNew[:, 0] = data[:, 0]
        data_new = np.array([[0, 0]])
        for data in data_blocks:
            start_index = 0
            while start_index < len(data)-1:
                end_index = find_nearest_index(data[:, 0], data[start_index][0]+window)
                segment = data[start_index:end_index]
                if len(segment) == 0:
                    start_index = find_nearest_index(data[:, 0], data[start_index, 0]+hop)
                    continue
                segment_clean = np.delete(segment, np.where(segment[:, 1] == -10000), axis=0)
                if len(segment_clean) == 0:
                    #After splitting into blocks, this loop better not come into play
                    #raise ValueError("This part of the block is absolute silence! Make sure block_thresh >= window!")
                    start_index = find_nearest_index(data[:, 0], data[start_index, 0]+hop)
                    continue
                n_clean = len(segment_clean)
                x_clean = np.matrix(segment_clean[:, 0]).reshape(n_clean, 1)
                y_clean = np.matrix(segment_clean[:, 1]).reshape(n_clean, 1)
                #return [x_clean, y_clean]
                theta = normal_equation(x_clean, y_clean)

                #determine the start and end of the segment to be labelled
                label_start_index = find_nearest_index(x_clean, data[start_index, 0]+label_start_offset)
                label_end_index = find_nearest_index(x_clean, data[start_index, 0]+label_end_offset)
                x_clean = x_clean[label_start_index:label_end_index]
                #return x_clean
                x_clean = np.insert(x_clean, 0, np.ones(len(x_clean)), axis=1)
                newy = x_clean*theta
                result = np.append(x_clean[:, 1], newy, axis=1)
                data_new = np.append(data_new, result, axis=0)

                start_index = find_nearest_index(data[:, 0], data[start_index, 0]+hop)

        return [data_new[:, 0], data_new[:, 1]]

################################################################################
# RECORDING                                                                    #
################################################################################

class Recording:
    def __init__(self, pitch_obj):
        self.pitch_obj = pitch_obj
        assert isinstance(self.pitch_obj, Pitch)
        self.histogram = None
        self.intonation_profile = None
        self.contour_labels = None

    def serialize_hist(self, path):
        pickle.dump(self.histogram, file(path, 'w'))

    def serialize_intonation(self, path):
        pickle.dump(self.intonation_profile, file(path, 'w'))

    def serialize_contour_labels(self, path):
        pickle.dump(self.contour_labels, file(path, 'w'))

    def compute_hist(self, bins=None, density=True, folded=False, weight="duration"):
        """
        Computes histogram from the pitch data in Pitch object (pitch), and creates
        a Data object (pypeaks).
        :param bins: Refers to number of bins in the histogram, determines the granularity.
        If it is not set, the number of bins which gives the highest granularity is chosen
        automatically.
        :param density: defaults to True, which means the histogram will be a normalized one.
        :param folded: defaults to False. When set to True, all the octaves are folded to one.
        :param weight: It can be one of the 'duration' or 'instance'. In the latter case, make
        sure that the pitch object has the pitch values discretized.
        """
        #Step 1: get the right pitch values
        assert isinstance(self.pitch_obj.pitch, np.ndarray)
        valid_pitch = self.pitch_obj.pitch
        valid_pitch = [i for i in valid_pitch if i > -10000]
        if folded:
            valid_pitch = map(lambda x: int(x % 1200), valid_pitch)
            valid_pitch = list(valid_pitch)

        #Step 2: based on the weighing scheme, compute the histogram
        if weight == "duration":
            #Step 2.1 set the number of bins (if not passed)
            if not bins:
                # CHANGE: added int
                bins = int(max(valid_pitch) - min(valid_pitch))
            n, bin_edges = np.histogram(valid_pitch, bins, density=density)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            self.histogram = Data(bin_centers, n)
        elif weight == "instance":
            n = {}
            i = 1
            while i < len(valid_pitch) - 1:
                if (valid_pitch[i] - valid_pitch[i - 1] != 0) and \
                        (valid_pitch[i + 1] - valid_pitch[i] == 0):
                    if valid_pitch[i] in n.keys():
                        n[valid_pitch[i]] += 1
                    else:
                        n[valid_pitch[i]] = 1
                i += 1
            n = n.items()
            n.sort(key=lambda x: x[0])
            n = np.array(n)
            self.histogram = Data(n[:, 0], n[:, 1])

            median_diff = np.median(np.diff(n[:, 0]))
            bin_edges = [n[0, 0] - median_diff/2]
            bin_edges.extend(median_diff/2 + n[:, 0])
            n[:, 1] = n[:, 1]/(n[:, 1].sum()*np.diff(bin_edges))
            self.histogram = Data(n[:, 0], n[:, 1], default_smooth=False)

    def parametrize_peaks(self, intervals, max_peakwidth=50, min_peakwidth=25, symmetric_bounds=True):
        """
        Computes and stores the intonation profile of an audio recording.
        :param intervals: these will be the reference set of intervals to which peak positions
         correspond to. For each interval, the properties of corresponding peak, if exists,
         will be computed and stored as intonation profile.
        :param max_peakwidth: the maximum allowed width of the peak at the base for computing
        parameters of the distribution.
        :param min_peakwidth: the minimum allowed width of the peak at the base for computing
        parameters of the distribution.
        """
        assert isinstance(self.pitch_obj.pitch, np.ndarray)
        valid_pitch = self.pitch_obj.pitch
        valid_pitch = [i for i in valid_pitch if i > -10000]
        valid_pitch = np.array(valid_pitch)

        parameters = {}
        for i in range(len(self.histogram.peaks["peaks"][0])):
            peak_pos = self.histogram.peaks["peaks"][0][i]
            #Set left and right bounds of the distribution.
            max_leftbound = peak_pos - max_peakwidth
            max_rightbound = peak_pos + max_peakwidth
            leftbound = max_leftbound
            rightbound = max_rightbound
            nearest_valleyindex = utils.find_nearest_index(self.histogram.peaks["valleys"][0], peak_pos)
            if peak_pos > self.histogram.peaks["valleys"][0][nearest_valleyindex]:
                leftbound = self.histogram.peaks["valleys"][0][nearest_valleyindex]
                if len(self.histogram.peaks["valleys"][0][nearest_valleyindex + 1:]) == 0:
                    rightbound = peak_pos + max_peakwidth
                else:
                    offset = nearest_valleyindex + 1
                    nearest_valleyindex = utils.find_nearest_index(
                        self.histogram.peaks["valleys"][0][offset:], peak_pos)
                    rightbound = self.histogram.peaks["valleys"][0][offset + nearest_valleyindex]
            else:
                rightbound = self.histogram.peaks["valleys"][0][nearest_valleyindex]
                if len(self.histogram.peaks["valleys"][0][:nearest_valleyindex]) == 0:
                    leftbound = peak_pos - max_peakwidth
                else:
                    nearest_valleyindex = utils.find_nearest_index(
                        self.histogram.peaks["valleys"][0][:nearest_valleyindex], peak_pos)
                    leftbound = self.histogram.peaks["valleys"][0][nearest_valleyindex]

            #In terms of x-axis, leftbound should be at least min_peakwidth
            # less than peak_pos, and at max max_peakwidth less than peak_pos,
            # and viceversa for the rightbound.
            if leftbound < max_leftbound:
                leftbound = max_leftbound
            elif leftbound > peak_pos - min_peakwidth:
                leftbound = peak_pos - min_peakwidth

            if rightbound > max_rightbound:
                rightbound = max_rightbound
            elif rightbound < peak_pos + min_peakwidth:
                rightbound = peak_pos + min_peakwidth

            #If symmetric bounds are asked for, then make the bounds symmetric
            if symmetric_bounds:
                if peak_pos - leftbound < rightbound - peak_pos:
                    imbalance = (rightbound - peak_pos) - (peak_pos - leftbound)
                    rightbound -= imbalance
                else:
                    imbalance = (peak_pos - leftbound) - (rightbound - peak_pos)
                    leftbound += imbalance

            #extract the distribution and estimate the parameters
            distribution = valid_pitch[valid_pitch >= leftbound]
            distribution = distribution[distribution <= rightbound]
            #print peak_pos, "\t", len(distribution), "\t", leftbound, "\t", rightbound

            interval_index = utils.find_nearest_index(intervals, peak_pos)
            interval = intervals[interval_index]
            _mean = float(np.mean(distribution))
            _variance = float(variation(distribution))
            _skew = float(skew(distribution))
            _kurtosis = float(kurtosis(distribution))
            pearson_skew = float(3.0 * (_mean - peak_pos) / np.sqrt(abs(_variance)))
            parameters[interval] = {"position": float(peak_pos),
                                    "mean": _mean,
                                    "amplitude": float(self.histogram.peaks["peaks"][1][i]),
                                    "variance": _variance,
                                    "skew1": _skew,
                                    "skew2": pearson_skew,
                                    "kurtosis": _kurtosis}

        self.intonation_profile = parameters

    def label_contours(self, intervals, window=150, hop=30):
        """
        In a very flowy contour, it is not trivial to say which pitch value corresponds
         to what interval. This function labels pitch contours with intervals by guessing
         from the characteristics of the contour and its melodic context.
        :param window: the size of window over which the context is gauged, in milliseconds.
        :param hop: hop size in milliseconds.
        """
        window /= 1000.0
        hop /= 1000.0
        exposure = int(window / hop)

        boundary = window - hop
        final_index = utils.find_nearest_index(self.pitch_obj.timestamps,
                                               self.pitch_obj.timestamps[-1] - boundary)

        interval = np.median(np.diff(self.pitch_obj.timestamps))
        #interval = 0.00290254832393
        window_step = window / interval
        hop_step = hop / interval
        start_index = 0
        end_index = window_step
        contour_labels = {}
        means = []
        while end_index < final_index:
            temp = self.pitch_obj.pitch[start_index:end_index][self.pitch_obj.pitch[start_index:end_index] > -10000]
            means.append(np.mean(temp))
            start_index = start_index + hop_step
            end_index = start_index + window_step

        for i in range(exposure, len(means) - exposure + 1):
            _median = np.median(means[i - exposure:i])
            if _median < -5000:
                continue
            ind = utils.find_nearest_index(_median, intervals)
            contour_end = (i - exposure) * hop_step + window_step
            contour_start = contour_end - hop_step
            #print sliceBegin, sliceEnd, JICents[ind]
            #newPitch[sliceBegin:sliceEnd] = JICents[ind]
            if intervals[ind] in contour_labels.keys():
                contour_labels[intervals[ind]].append([contour_start, contour_end])
            else:
                contour_labels[intervals[ind]] = [[contour_start, contour_end]]

        self.contour_labels = contour_labels

    def plot_contour_labels(self, new_fig=True):
        """
        Plots the labelled contours!
        """
        timestamps = []
        pitch = []

        if new_fig:
            p.figure()
        for interval, contours in self.contour_labels.items():
            for contour in contours:
                x = self.pitch_obj.timestamps[contour[0]:contour[1]]
                y = [interval]*len(x)
                timestamps.extend(x)
                pitch.extend(y)

        data = np.array([timestamps, pitch]).T
        data = np.array(sorted(data, key=lambda xx: xx[0]))
        p.plot(data[:, 0], data[:, 1], 'g-')

################################################################################
# UTILS                                                                        #
################################################################################
def find_nearest_index(arr, value):
    """For a given value, the function finds the nearest value
    in the array and returns its index."""
    arr = np.array(arr)
    index = (abs(arr-value)).argmin()
    return index

def normal_equation(x, y):
    """
    X: matrix of features, one sample per row (without bias unit)
    y: values (continuous) corresponding to rows (samples) in X
    """
    num_samples = y.size
    x = np.insert(x, 0, np.ones(num_samples), axis=1)

    return pinv(x)*y

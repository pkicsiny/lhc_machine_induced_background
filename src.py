import numpy as np
from ipywidgets import interact, IntSlider


def readH5Data(hf_files_list, disc_list=[4, 4], ring_list=[1, 1], group_name='Tof_q', row_size=2):
    """
    Reads hdf5 data from a list of files. 
    Reads specific datasets only, defined by disc_list and ring_list.
    Optimized for hdf5 files with dataspace shape of (N, row_size) corresponding to time of flight - charge.
    The lists hf_files_list, disc_list and ring_list must have the same length.
    :param hf_files_list: list of h5py._hl.files.File types. Hdf5 files to read.
    :param disc_list: list of integers or list of lists of integers specifying TEPX discs
    :param ring_list: list of integers or list of lists of integers specifying TEPX rings
    :param group_name: string, hdf5 group name to read
    :param row_size: int, size of one data tuple in the hdf5
    :return:
        hf_sets: list of numpy arrays of shape (N, 2) where N is the number of tof-Q entriy pairs.
    """
    
    # list of return data
    hf_dsets = []
    
    # loop over hdf5 files
    for hf_file, disc, ring in zip(hf_files_list, disc_list, ring_list):
    
        print("Reading hdf5 data...")
        
        # check if hf_file is a list of files or a single file
        if type(hf_file) != list:
            
            # listify file for ease
            hf_file = [hf_file]
        
        # init data container
        hf_dset = np.empty((0,row_size))
            
        # loop over listified list of file(s)
        for hfile in hf_file:
            
            print("Reading: {}".format(hfile))
                
            #get data group
            hf_group = hfile.get(group_name)
            
            # loop over discs and rings that need to be read
            if type(disc) != list:
                disc = [disc]
            if type(ring) != list:
                ring = [ring]
            for d in disc:
                for r in ring:
                    
                    # init data key
                    disc_ring = "d{}r{}".format(d, r)
    
                    # [0, 0] entries are empty, and are there just because of the placeholder, need to be deleted
                    for key in hf_group.keys():
                
                        # get right key
                        if disc_ring in key:
                            hf_ptype = hf_group.get(key)
                    
                            # check if data is empty else read it and append to dset
                            if hf_ptype[:].shape == (1,2) and all(i == 0 for i in hf_ptype[0]):
                                print("    0 entries for {}".format(key))
                                continue
                            else:
                                print("    {} entries for {}".format(len(hf_ptype[:]), key))
                                hf_dset = np.vstack((hf_dset, hf_ptype[:]))
                
        print("Number of entries read: ", len(hf_dset))
        hf_dsets.append(hf_dset)
        
    return hf_dsets


def binH5Data(hf_dset, n_bins=100, chunk_size=10, tof_scaling=1, q_scaling=1, q_threshold=0, shift_n_times=0, shift_offset=0, xedges=None, yedges=None, epsilon=1e-8, verbose=True):
    """
    Bins data into a 2D array.
    :param hf_dset: numpy array of size(N, 2) where N is the number of tof-Q entry pairs.
    :param chunk_size: int, the data will be read and pre-processed in chunks to save memory.
    :param tof_scaling: float, scaling on the horizontal axis e.g. from units of nanoseconds to seconds use 1e-9
    :param q_scaling: float, scaling on the vertical axis e.g. from GeV to eV use 1e9, from eV to e- use 1/3.6eV (in Si)
    :param q_threshold: float, data with charge below (<) this value will be ignored and filled in a separate histogram
    :param shift_n_times: int, duplicate data along the horizontal axis this many times, in both directions
    :param shift_offset: float, distance between 2 data duplicates along horizontal axis. Has to be in the same units as the tof data.
    :param xedges: None or numpy array of size (n_bins+1,), tof histogram x axis bin limits. If given it will be used,
    if set to None it will be determined from the data set.
    :param yedges: None or numpy array of size (n_bins+1,), tof histogram y axis bin limits. If given it will be used,
    if set to None it will be determined from the data set.
    :param epsilon: float, for handling numerical errors
    :param verbose: bool, print more info during run
    :return:
        tof_histo: numpy array of size (n_bins, n_bins), contains binned tof-Q data
        tof_below_histo: numpy array of size (n_bins, n_bins), contains binned tof-Q data below q_threshold
        tof_above_histo: numpy array of size (n_bins, n_bins), contains binned tof-Q data above q_threshold
        xedges: numpy array of size (n_bins+1,), tof histogram x axis bin limits. Same for all 3 returned histograms.
        yedges: numpy array of size (n_bins+1,), tof histogram y axis bin limits. Same for all 3 returned histograms.
    """
   
    # chunks size to read data in chunk_size chunks
    n_chunks = int(np.ceil(len(hf_dset)/chunk_size))
 
    #find the overall min/max
    if xedges is None:
        tof_low = np.inf
        tof_high = -np.inf
    if yedges is None:
        q_low = np.inf
        q_high = -np.inf   
    
    # x axis in [ns] (*1e-9) y axis in [e-]
    if xedges is None or yedges is None:
        for chunk_idx in range(n_chunks):
            print("Reading chunk to examine data scale [{}/{}]".format(chunk_idx+1, n_chunks))
            if verbose:
                print("tof min: {}, tof max: {}".format(tof_low, tof_high))
                print("q min: {}, q max: {}".format(q_low, q_high))
            if xedges is None:
                tof_low = np.minimum(hf_dset[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, 0].min(), tof_low)
                tof_high = np.maximum(hf_dset[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, 0].max(), tof_high)
            if yedges is None:
                q_low = np.minimum(hf_dset[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, 1].min(), q_low)
                q_high = np.maximum(hf_dset[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, 1].max(), q_high)
        
        # extend tof range to copy data
        if xedges is None:
            tof_high += shift_n_times*shift_offset
            tof_low -= shift_n_times*shift_offset
    
        # scale bin limits
        if xedges is None:
            tof_low = tof_low*tof_scaling
            tof_high = tof_high*tof_scaling
        if yedges is None:
            q_low = q_low*q_scaling
            q_high = q_high*q_scaling
    
    # create bin limits based on min/max values, and scale the values
    tof_bin_edges = np.linspace(tof_low-epsilon, tof_high+epsilon, (2*shift_n_times+1)*n_bins+1) if xedges is None else xedges
    q_bin_edges = np.linspace(q_low-epsilon, q_high+epsilon, n_bins+1) if yedges is None else yedges
    
    #create empty 2d histograms
    tof_histo = np.zeros((n_bins, (2*shift_n_times+1)*n_bins))
    tof_below_histo = np.zeros((n_bins, (2*shift_n_times+1)*n_bins))
    tof_above_histo = np.zeros((n_bins, (2*shift_n_times+1)*n_bins))
    
    # iterate over the dataset in chunks of n_chunks lines
    for chunk_idx in range(n_chunks):
        print("Reading chunk to bin data [{}/{}]".format(chunk_idx+1, n_chunks))
        
        # x axis in [ns] (*1e-9) y axis in [e-]
        tof_chunk = hf_dset[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, 0]
        q_chunk = hf_dset[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size, 1]
        if verbose:
            print("New data chunk length: {}".format(len(tof_chunk)))
        
        # create data copies (offset value in [ns])
        tof_chunk_extended = np.copy(tof_chunk)
        q_chunk_extended = np.copy(q_chunk)
        for i in range(1, shift_n_times+1):
            tof_chunk_extended = np.concatenate((tof_chunk_extended, tof_chunk+i*shift_offset, tof_chunk-i*shift_offset))
            q_chunk_extended = np.concatenate((q_chunk_extended, q_chunk, q_chunk))    
        tof_chunk = tof_chunk_extended
        q_chunk = q_chunk_extended
        if verbose:
            print("Extended data chunk length: {}".format(len(tof_chunk)))
        
        # scale data
        tof_chunk *= tof_scaling
        q_chunk *= q_scaling
        
        # split data along q threshold
        tof_chunk_below = tof_chunk[q_chunk < q_threshold]
        q_chunk_below = q_chunk[q_chunk < q_threshold]
        tof_chunk_above = tof_chunk[q_chunk >= q_threshold]
        q_chunk_above = q_chunk[q_chunk >= q_threshold]
        
        #fill 2d histo with data chunk
        tof_chunk_histo, xedges, yedges = np.histogram2d(
            tof_chunk, q_chunk, bins=(tof_bin_edges, q_bin_edges));
        tof_chunk_below_histo, xedges_below, yedges_below = np.histogram2d(
            tof_chunk_below, q_chunk_below, bins=(tof_bin_edges, q_bin_edges));
        tof_chunk_above_histo, xedges_above, yedges_above = np.histogram2d(
            tof_chunk_above, q_chunk_above, bins=(tof_bin_edges, q_bin_edges));
        
        # accumulate bin counts over chunks
        tof_histo += tof_chunk_histo.T
        tof_below_histo += tof_chunk_below_histo.T
        tof_above_histo += tof_chunk_above_histo.T
        print("Hits added to histo so far: {}".format(int(np.sum(tof_histo))))
    
    return tof_histo, tof_below_histo, tof_above_histo, xedges, yedges


def getAreaBelowCurve(xmin, xmax, ymin, ymax, cx, cy, curve_x, curve_y, epsilon=1e-8, num_precision=8, verbose=False):
    """
    Calculates the absolute area of a cell below a curve defined by curve_x and curve_y.
    -finds the curve segment closest to the cell
    -checks if there are multiple segments inside the cell
    -divides cell with vertical lines into #segments_inside_cell many subcells
    -calculates area below the corresponding curve segment for each subcell
    -sums up subcell areas

    :params xmin, xmax, ymin, ymax: Float coordinates of bin cell corners
    :params cx, cy: Float coordinates of bin cell center
    :params curve_x, curve_y: array or pandas dataframe of Floats containing mask curve coordinates
    :param epsilon: Float, used to prevent misassignment errors due to numerical precision
    :param num_precision: int to round all parameters to this many decimals
    :param verbose: bool, print info if set to True
    :return: area_below_curve
    """
    
    # round for safety reasons
    xmin = round(xmin, num_precision)
    xmax = round(xmax, num_precision)
    ymin = round(ymin, num_precision)
    ymax = round(ymax, num_precision)
    cx = round(cx, num_precision)
    cy = round(cy, num_precision)
    curve_x = np.array([round(xi, num_precision) for xi in curve_x])
    curve_y = np.array([round(yi, num_precision) for yi in curve_y])
    
    if verbose:
        print("getAreaBelowCurve: current cell center: x: {}, y: {}".format(cx, cy))
    
    # upper segment endpoint index can be at most the last point on curve and at least index 1
    try:
        p2x_idx = np.maximum(int(np.argwhere(np.array(curve_x) > cx)[0]), 1)
    
    #if cell center is to the right of the whole curve,
    # take (last occurence, relevant if line segment is vertical) arg of maximum x coordinate
    except:
        p2x_idx = len(curve_x) - np.argmax(curve_x[::-1]) - 1 
    
    #lower segment endpoint index can be at least index 0 and at most the second last point on curve
    p1x_idx = np.maximum(p2x_idx - 1, 0)
    
    #line segment closest to cell center (based on x coordinate)
    pmin, pmax = curve_x[p1x_idx:p2x_idx+1]
    
    if verbose:
        print("getAreaBelowCurve: curve segment indices closest to cell center: p1x_idx: {}, p2x_idx: {}".format(p1x_idx, p2x_idx))
        print("getAreaBelowCurve: curve segment x coordinates closest to cell center: p1x: {}, p2x: {}".format(pmin, pmax))
        
    #if any line segment endpoints are inside the cell x range, include next endpoint until it is outside
    while pmin > xmin:
        if p1x_idx == 0 or np.abs((pmin-xmin)/xmin) < epsilon:
            break
        else:
            p1x_idx -= 1
            pmin = curve_x[p1x_idx]
    while pmax < xmax:
        if p2x_idx == len(curve_x) - np.argmax(curve_x[::-1]) - 1 or np.abs((pmax-xmax)/xmax) < epsilon:
            break
        else:
            p2x_idx += 1
            pmax = curve_x[p2x_idx]
            
    if verbose:
        print("getAreaBelowCurve: range of curve segment indices inside cell x range: p1x_idx: {}, p2x_idx: {}".format(p1x_idx, p2x_idx))
        print("getAreaBelowCurve: range of curve segment x coordinates inside cell x range: p1x: {}, p2x: {}".format(pmin, pmax))
        print("getAreaBelowCurve: number of subcells: {}".format(p2x_idx-p1x_idx))
            
    # divide the cell with vertical lines according to the number of line segments inside the cell
    area_below_curve_subcells = []
    for i in range(p1x_idx, p2x_idx):
        
        if verbose:
            print("getAreaBelowCurve: calculating area for (sub)cell under curve segment {}-{}".format(i, i+1))
        
        # if the bottom segment endpoint is the absolute first point, use the cell xmin in any case
        # (works only if first segment is horizontal)
        if i == p1x_idx:
            sub_xmin = xmin
        else:
            sub_xmin = curve_x[i]
    
        # if the top segment endpoint is the absolute last point, use the cell xmax in any case  
        # (good if last segment is vertical)
        if i+1 == p2x_idx:
            sub_xmax = xmax
        else:  
            sub_xmax = curve_x[i+1]
            
        #select closest line segment to subcell 
        #p1x and p2x are the same as pmin and pmax if cell x range is within one line segment
        p1x, p1y = curve_x[i], curve_y[i]
        p2x, p2y = curve_x[i+1], curve_y[i+1]
        
        # subcell crossings with curve segment
        p_left, p_right, p_top, p_bottom = getCrossingPoints(
                p1x, p1y, p2x, p2y, sub_xmin, sub_xmax, ymin, ymax, epsilon=1e-16, num_precision=num_precision, verbose=verbose)
    
        # get subcell area below curve (give subcell center x coordinate to function)
        sub_cx = round((sub_xmax + sub_xmin)/2, num_precision)
        area_below_curve_subcells.append(getSingleCellMaskValue(
                p_left, p_right, p_top, p_bottom, sub_xmin, sub_xmax, ymin, ymax,
                sub_cx, cy, p1x, p1y, p2x, p2y, num_precision=num_precision, verbose=verbose))
        
    # sum up subcell areas below line segment to get area for full cell
    area_below_curve = round(np.sum(area_below_curve_subcells), num_precision)
    
    if verbose:
        print("getAreaBelowCurve: subcell areas: {} and their sum: {} (full bin cell area: {})".format(area_below_curve_subcells, area_below_curve, round((xmax-xmin)*(ymax-ymin), num_precision)))
    
    return area_below_curve


"""
assumes p1x <= p2x and p1y <= p2y (monotonically inreasing curve)
:param x: tuple or list or array of Float of shape 1x2 defining the point in question
:params p1, p2: tuples or lists or arrays of Float of shape 1x2
defining coordinates of line segment endpoints
:return: 1 if point x is ABOVE p1-p2 line, -1 if BELOW and 0 if right ON it
"""
isabove = lambda x, p1,p2: -1 if np.cross(np.array(p2)-np.array(p1), np.array(x)-np.array(p1)) < 0 else \
                            1 if np.cross(np.array(p2)-np.array(p1), np.array(x)-np.array(p1)) > 0 else 0


def getCrossingPoints(p1x, p1y, p2x, p2y, xmin, xmax, ymin, ymax, epsilon=1e-16, num_precision=8, verbose=False):
    """
    Calculates crossing point coordinates between (infinite) line segment and (sub-)cell sides.
    Returns 4 numpy arrays of shape 1x2 Float or None if the line does not cross that side.
    
    :params p1x, p1y, p2x, p2y: Float coordinates of line segment endpoints
    :params xmin, xmax, ymin, ymax: Float coordinates of bin cell corners
    :params epsilon: used fo avoid division by 0.
    Here the epsilon should be even smaller as slopes are typically O(1e-8)
    :param num_precision: int to round all parameters to this many decimals
    :param verbose: bool, print info if set to True
    :return: left, right, top, bottom
    """
    
    # round for safety reasons
    p1x = round(p1x, num_precision)
    p1y = round(p1y, num_precision)
    p2x = round(p2x, num_precision)
    p2y = round(p2y, num_precision)
    xmin = round(xmin, num_precision)
    xmax = round(xmax, num_precision)
    ymin = round(ymin, num_precision)
    ymax = round(ymax, num_precision)
    
    #initialize crossing point coordinate arrays
    left = np.array([None, None])
    right = np.array([None, None])
    top = np.array([None, None])
    bottom = np.array([None, None])
    num_crossing_points = 0
    
    #slope of line segment
    m = round((p2y - p1y)/(p2x - p1x + epsilon), num_precision)
    
    if verbose:
        print("getCrossingPoints: curve segment slope: {}".format(m))
    
    # need to correct for infinite slope (vertical line)
    
    # crosses left side
    y = round(m * (xmin - p1x) + p1y, num_precision)
    if (y >= ymin and y <= ymax): 
        left = np.array([xmin, y])
        num_crossing_points += 1

    # crosses right side
    y = round(m * (xmax - p1x) + p1y, num_precision)
    if (y >= ymin and y <= ymax): 
        right = np.array([xmax, y])
        num_crossing_points += 1
        
    # crosses bottom side
    x = round((ymin - p1y)/(m + epsilon) + p1x, num_precision)
    if (x >= xmin and x <= xmax):
        bottom =  np.array([x, ymin])
        num_crossing_points += 1
        
    # crosses top side
    x = round((ymax - p1y) /(m + epsilon) + p1x, num_precision)
    if (x >= xmin and x <= xmax):
        top = np.array([x, ymax])
        num_crossing_points += 1
    
    if verbose:
        print("getCrossingPoints: crossing point coordinates (x,y): left: {}, right: {}, top: {}, bottom: {}".format(left, right, top, bottom))
        if num_crossing_points!=2 and num_crossing_points!=0:
            print("\x1b[31mgetCrossingPoints: warning in calculating crossing points. Two or zero crossings expected but {} obtained. OK if line segment ends parallel to cell edge or it crosses through (sub)cell corner.\x1b[0m".format(num_crossing_points))
        
  #  assert num_crossing_points==2 or num_crossing_points==0, "getCrossingPoints: error in calculating crossing points. Two or zero crossings expected but {} obtained.".format(num_crossing_points)
        
    return left, right, top, bottom


def getSingleCellMaskValue(left, right, top, bottom, xmin, xmax, ymin, ymax,
                           cx, cy, p1x, p1y, p2x, p2y, num_precision=8, verbose=False):
    """  
    Monotonically increasing curve can cross a rectangular cell in 4 ways:
    bottom in right out
    bottom in top out
    left in right out
    left in top out
    
    Calculates cell area below a line segment.
    Returns the area BELOW the line segment, total cell area and the 
    quotient of these which is the normalized area that is used as mask value,
    and the cell center coordinates.
    
    :params left, right, top, bottom: numpy arrays of Float or None of shape 1x2
    containing line segment and cell side crossing points at respective sides
    :params xmin, xmax, ymin, ymax: Float coordinates of bin cell corners
    :params cx, cy: Float coordinates of bin cell center
    :params p1x, p1y, p2x, p2y: Float coordinates of line segment endpoints
    :param num_precision: int to round all parameters to this many decimals
    :param verbose: bool, print info if set to True
    :return: area_below_curve, cell_mask_value, cy, cy
    """
    
    # round for safety reasons
    p1x = round(p1x, num_precision)
    p1y = round(p1y, num_precision)
    p2x = round(p2x, num_precision)
    p2y = round(p2y, num_precision)
    cx = round(cx, num_precision)
    cy = round(cy, num_precision)
    xmin = round(xmin, num_precision)
    xmax = round(xmax, num_precision)
    ymin = round(ymin, num_precision)
    ymax = round(ymax, num_precision)
    
    #initialize return values
    area_below_curve = 0
    cell_mask_value = 0

    #line segment does not cross cell
    if (None in left) and (None in right) and (None in top) and (None in bottom):
        if verbose:
            print("getSingleCellMaskValue: curve segment ({}, {})-({}, {}) does not cross (sub)cell".format(p1x, p1y, p2x, p2y))
        
        #cell is fully above line segment
        orientation = isabove((cx, cy), (p1x, p1y), (p2x, p2y))
        
        if verbose:
            
                # 2D cross product (3D but Z=0)
                cross_product = round(float(np.cross(np.array((p2x, p2y))-np.array((p1x, p1y)),np.array((cx, cy))-np.array((p1x, p1y)))), num_precision)
                print("getSingleCellMaskValue: cross product: ", cross_product)
                print("getSingleCellMaskValue: (sub)cell center, left endpoint, right endpoint: ", (cx, cy), (p1x, p1y), (p2x, p2y))
        
        if orientation == 1:
            if verbose:
                print("getSingleCellMaskValue: (sub)cell is above curve segment ({}, {})-({}, {})".format(p1x, p1y, p2x, p2y))
            area_below_curve = 0
        
        #cell is fully under line segment
        elif orientation == -1:
            if verbose:
                print("getSingleCellMaskValue: (sub)cell is below curve segment ({}, {})-({}, {})".format(p1x, p1y, p2x, p2y))
            area_below_curve = round((xmax - xmin)*(ymax - ymin), num_precision)
        
        #cell center lies on line segment but line segment does not cross cell (impossible)
        else:
            raise RuntimeError("getSingleCellMaskValue: (sub)cell center lies on line segment ({}, {})-({}, {}) but cell is not crossed. Check inputs!".format(p1x, p1y, p2x, p2y))
            
    #line segment crosses cell
    else:
        if verbose:
            print("getSingleCellMaskValue: curve segment ({}, {})-({}, {}) crosses (sub)cell".format(p1x, p1y, p2x, p2y))
        
        #bottom in right out
        if (None not in bottom) and (None not in right):
            if verbose:
                print("getSingleCellMaskValue: bottom in right out")
            
            #a*b/2
            area_below_curve = round((xmax - bottom[0])*(right[1] - ymin)/2, num_precision)
            
        #bottom in top out
        if (None not in bottom) and (None not in top):
            if verbose:
                print("getSingleCellMaskValue: bottom in top out")
            
            #b*(a+c)/2
            area_below_curve = round((ymax - ymin)*(2*xmax - bottom[0] - top[0])/2, num_precision)
            
        #left in right out
        if (None not in left) and (None not in right):
            if verbose:
                print("getSingleCellMaskValue: left in right out")
            
            #a*(b+d)/2
            area_below_curve = round((xmax - xmin)*(left[1] + right[1] - 2*ymin)/2, num_precision)
            
        #left in top out
        if (None not in left) and (None not in top):
            if verbose:
                print("getSingleCellMaskValue: left in top out")
            
            #x*y-a*b/2
            area_below_curve = round((xmax - xmin)*(ymax - ymin) - (top[0] - xmin)*(ymax - left[1])/2, num_precision)
       
    if verbose:
        print("getSingleCellMaskValue: (sub)cell area below curve segment ({}, {})-({}, {}): {}".format(p1x, p1y, p2x, p2y, area_below_curve))
    return area_below_curve


def mcOverlay(mask, data):
    """
    Masks *data* with *mask* such that only integer values are allowed.
    :param mask: 2D numpy array of Floats
    :param data: 2D numpy array of Floats, same size as *mask*
    :return: masked_data
    """
    
    masked_data = np.zeros_like(data)
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            masked_data[i, j] = np.sum(np.random.rand(int(data[i, j])) < mask[i, j])
    return masked_data


def freeze_header(df, num_rows=30, num_columns=12, step_rows=1, step_columns=1):
    """
    idea: https://stackoverflow.com/questions/28778668/freeze-header-in-pandas-dataframe
    Freeze the headers (column and index names) of a Pandas DataFrame. A widget
    enables to slide through the rows and columns.
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame to display
    num_rows : int, optional
        Number of rows to display
    num_columns : int, optional
        Number of columns to display
    step_rows : int, optional
        Step in the rows
    step_columns : int, optional
        Step in the columns
    Returns
    -------
    Displays the DataFrame with the widget
    """

    @interact(last_row=IntSlider(min=min(num_rows, df.shape[0]),
                                 max=df.shape[0],
                                 step=step_rows,
                                 description='rows',
                                 readout=False,
                                 disabled=False,
                                 continuous_update=True,
                                 orientation='horizontal',
                                 slider_color='purple'),
              last_column=IntSlider(min=min(num_columns, df.shape[1]),
                                    max=df.shape[1],
                                    step=step_columns,
                                    description='columns',
                                    readout=False,
                                    disabled=False,
                                    continuous_update=True,
                                    orientation='horizontal',
                                    slider_color='purple'))

    def _freeze_header(last_row, last_column):
        display(df.iloc[max(0, last_row - num_rows):last_row,
                max(0, last_column - num_columns):last_column])
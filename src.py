import numpy as np
from ipywidgets import interact, IntSlider

def getAreaBelowCurve(xmin, xmax, ymin, ymax, cx, cy, curve_x, curve_y, epsilon=1e-8, verbose=False):
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
    :param verbose: bool, print info if set to True
    :return: area_below_curve
    """
    
    if verbose:
        print("current cell center: x: {}, y: {}".format(cx, cy))
    
    # upper segment endpoint index can be at most the last point on curve and at least index 1
    try:
        p2x_idx = np.maximum(int(np.argwhere(np.array(curve_x) > cx)[0]), 1)
    
    #if cell center is to the right of the whole curve,
    # take (last occurence, relevant if line segment is vertical) arg of maximum x coordinate
    except:
        p2x_idx = len(curve_x) - np.argmax(curve_x[::-1]) - 1 
    
    #lower segment endpoint index can be at least index 0 and at most the second last point on curve
    p1x_idx = np.maximum(p2x_idx - 1, 0)
    
    #line segment closest to cell center
    pmin, pmax = curve_x[p1x_idx:p2x_idx+1]
    if verbose:
        print("curve segment indices closest to cell center: p1x_idx: {}, p2x_idx: {}".format(p1x_idx, p2x_idx))
        print("curve segment x coordinates closest to cell center: p1x: {}, p2x: {}".format(pmin, pmax))
        
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
        print("range of curve segment indices inside cell x range: p1x_idx: {}, p2x_idx: {}".format(p1x_idx, p2x_idx))
        print("curve segment x coordinates inside cell x range: p1x: {}, p2x: {}".format(pmin, pmax))
        print("number of subcells: {}".format(p2x_idx-p1x_idx))
            
    # divide the cell with vertical lines according to the number of line segments inside the cell
    area_below_curve_subcells = []
    for i in range(p1x_idx, p2x_idx):
        
        # if the bottom segment endpoint is the absolute first point, use the cell xmin in any case
        # (good if first segment is horizontal)
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
                p1x, p1y, p2x, p2y, sub_xmin, sub_xmax, ymin, ymax, epsilon=1e-16)
    
        # get subcell area below curve (give subcell center x coordinate to function)
        area_below_curve_subcells.append(getSingleCellMaskValue(
                p_left, p_right, p_top, p_bottom, sub_xmin, sub_xmax, ymin, ymax,
                (sub_xmax + sub_xmin)/2, cy, p1x, p1y, p2x, p2y, verbose=verbose))
        
    # sum up subcell areas below line segment to get area for full cell
    area_below_curve = np.sum(area_below_curve_subcells)
    if verbose:
        print("subcell areas: {} and their sum: {}".format(area_below_curve_subcells, area_below_curve))
    
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


def getCrossingPoints(p1x, p1y, p2x, p2y, xmin, xmax, ymin, ymax, epsilon=1e-16):
    """
    Calculates crossing point coordinates between (infinite) line segment and (sub-)cell sides.
    Returns 4 numpy arrays of shape 1x2 Float or None if the line does not cross that side.
    
    :params p1x, p1y, p2x, p2y: Float coordinates of line segment endpoints
    :params xmin, xmax, ymin, ymax: Float coordinates of bin cell corners
    :params epsilon: used fo avoid division by 0.
    Here the epsilon should be even smaller as slopes are typically O(1e-8)
    :return: left, right, top, bottom
    """
    
    #initialize crossing point coordinate arrays
    left = np.array([None, None])
    right = np.array([None, None])
    top = np.array([None, None])
    bottom = np.array([None, None])
    
    #slope of line segment
    m = (p2y - p1y)/(p2x - p1x + epsilon)
    
    # crosses left side
    y = m * (xmin - p1x) + p1y 
    if (y >= ymin and y <= ymax): 
        left = np.array([xmin, y])

    # crosses right side
    y = m * (xmax - p1x) + p1y
    if (y >= ymin and y <= ymax): 
        right = np.array([xmax, y])
        
    # crosses bottom side
    x = (ymin - p1y)/(m + epsilon) + p1x
    if (x >= xmin and x <= xmax):
        bottom =  np.array([x, ymin])
        
    # crosses top side
    x = (ymax - p1y) /(m + epsilon) + p1x 
    if (x >= xmin and x <= xmax):
        top = np.array([x, ymax])
        
    return left, right, top, bottom


def getSingleCellMaskValue(left, right, top, bottom, xmin, xmax, ymin, ymax, cx, cy, p1x, p1y, p2x, p2y, verbose=False):
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
    :param verbose: bool, print info if set to True
    :return: area_below_curve, cell_mask_value, cy, cy
    """
    
    #initialize return values
    area_below_curve = 0
    cell_mask_value = 0

    #line segment does not cross cell
    if not left.all() and not right.all() and not top.all() and not bottom.all():
        if verbose:
            print("curve segment does not cross cell")
        
        #cell is fully above line segment
        orientation = isabove((cx, cy), (p1x, p1y), (p2x, p2y))
        if verbose:
                print("cross product: ", np.cross(np.array((p2x, p2y))-np.array((p1x, p1y)),
                                                  np.array((cx, cy))-np.array((p1x, p1y))))
                print("cell center, left endpoint, right endpoint: ", (cx, cy), (p1x, p1y), (p2x, p2y))
        
        if orientation == 1:
            if verbose:
                print("cell is above curve segment")
            area_below_curve = 0
        
        #cell is fully under line segment
        elif orientation == -1:
            if verbose:
                print("cell is below curve segment")
            area_below_curve = (xmax - xmin)*(ymax - ymin)
        
        #cell center lies on line segment but line segment does not cross cell (impossible)
        else:
            raise RuntimeError("Bin center lies on line segment but cell is not crossed. Check inputs!")
            
    #line segment crosses cell
    else:
        if verbose:
            print("curve segment crosses cell")
        
        #bottom in right out
        if bottom.all() and right.all():
            if verbose:
                print("bottom in right out")
            
            #a*b/2
            area_below_curve = (xmax - bottom[0])*(right[1] - ymin)/2
            
        #bottom in top out
        if bottom.all() and top.all():
            if verbose:
                print("bottom in top out")
            
            #b*(a+c)/2
            area_below_curve = (ymax - ymin)*(2*xmax - bottom[0] - top[0])/2
            
        #left in right out
        if left.all() and right.all():
            if verbose:
                print("left in right out")
            
            #a*(b+d)/2
            area_below_curve = (xmax - xmin)*(left[1] + right[1] - 2*ymin)/2
            
        #left in top out
        if left.all() and top.all():
            if verbose:
                print("left in top out")
            
            #x*y-a*b/2
            area_below_curve = (xmax - xmin)*(ymax - ymin) - (top[0] - xmin)*(ymax - left[1])/2
        
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
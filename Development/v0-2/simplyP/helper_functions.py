"""
Helper functions for common procedures such as unit conversions
"""

# Convenience functions for unit conversions
def UC_Q(Q_mmd, A_catch):
    """ Convert discharge from units of mm/day to m3/day

    Args:
        Q_mmd:   Float. Discharge in mm/day
        A_catch: Float. Catchment area in km2

    Returns:
        Float. Discharge in m3/day
    """
    Q_m3d = Q_mmd*1000*A_catch
    return Q_m3d

def UC_Qinv(Q_m3s, A_catch):
    """ Convert discharge from units of m3/s to mm/day

    Args:
        Q_m3s:   Float. Discharge in m3/s
        A_catch: Float. Catchment area in km2

    Returns:
        Float. Discharge in mm/day
    """
    Q_mmd = Q_m3s * 86400/(1000*A_catch)
    return Q_mmd

def UC_C(C_kgmm, A_catch):
    """ Convert concentration from units of kg/mm to mg/l.
        Divide answer by 10**6 to convert from mg/mm to mg/l

    Args:
        C_kgmm:  Float. Concentration in kg/mm
        A_catch: Float. Catchment area in km2

    Returns:
        Float. Concentration in mg/l
    """    
    C_mgl = C_kgmm/A_catch
    return C_mgl

def UC_Cinv(C_mgl, A_catch):
    """ Convert concentration from units of mg/l to kg/mm

    Args:
        C_mgl:   Float. Concentration in mg/l
        A_catch: Float. Catchment area in km2

    Returns:
        Float. Concentration in kg/mm
    """ 
    C_kgmm = C_mgl*A_catch
    return C_kgmm
    
def UC_V(V_mm, A_catch, outUnits):
    """ Convert volume from mm to m^3 or litres

    Args:
        V_mm:     Float. Depth in mm
        outUnits: Str. 'm3' or 'l'
        A_catch:  Float. Catchment area in km2

    Returns:
        Float. Volume in specified units.
    """
    factorDict = {'m3':10**3, 'l':10**6}
    V = V_mm * factorDict[outUnits] * A_catch
    return V

# ###################################################################################
# Other helper functions

def lin_interp(x, x0, x1, y0, y1):
    """ Simple helper function for linear interpolation. Estimates the value of y at x
        by linearly interpolating between points (x0, y0) and (x1, y1), where x0 <= x < x1.

    Args:
        x:  Float. x-value for which to esrimate y
        x0: Float. x-value of point 0
        x1: Float. x-value of point 1 
        y0: Float. y-value of point 0
        y1: Float. y-value of point 1

    Returns:
        Float. Estimated value of y at x.
    """
    y = y0 + (y1-y0)*(x-x0)/(x1-x0)
    
    return y
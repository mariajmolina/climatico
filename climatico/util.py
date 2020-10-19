def pacific_lon(longitude, to180=True):
    """
    Help converting Pacific longitudes to 180 or 360 coordinates.
    
    Args:
        longitude (float): Single longitude value.
        to180 (boolean): If coords are in 360 degrees. Defaults to ``True``.
    """
    if to180:
        if longitude <= 180:
            return longitude
        if longitude > 180:
            return longitude-360
    if not to180:
        if longitude < 0:
            return longitude+360
        if longitude >= 0:
            return longitude

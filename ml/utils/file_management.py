def check_savefile_integrity(filename):
    """
    Check if savefile of model (gen/disc/classifier) is valid
    :param filename: name of save file
    :return: bool value indicating if file is valid
    """
    if filename and not (filename.endswith('.pt') or filename.endswith('.pth')):
        return False
    return True

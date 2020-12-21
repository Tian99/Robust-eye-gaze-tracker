from PyQt5.QtWidgets import QMessageBox


def mkmessage(msgtext):
    """quick message in a box
    useful for warning user about bad input
    @param msgtext text to display in messagebox"""
    msg = QMessageBox()
    msg.setText(msgtext)
    msg.show()
    return msg.exec_()


def get_ROI(cropping_factor):
    '''
    This one simply cleaer the formatting issues for the cropping factors on the image
    Another one is CPI which stores [x1, y1, x2, y2]
    @param cropping_factor [[x1,x2], [y1, y2]]
    @return ROI like  [x, y, x_displacement, y_displacement]
    >>> get_ROI([[1, 3], [1, 5]])
    (1, 1, 2, 4)
    '''
    return (cropping_factor[0][0],
            cropping_factor[1][0],
            cropping_factor[0][1] - cropping_factor[0][0],
            cropping_factor[1][1] - cropping_factor[1][0])


def get_center(ROI):
    '''
    center coordinate using ROI croppig factor
    >>> get_center([0, 0, 4, 2])
    (2.0, 1.0)
    '''
    return (ROI[0] + ROI[2]/2, ROI[1] + ROI[3]/2)

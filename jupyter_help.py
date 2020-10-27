from PIL import Image
from IPython.display import display
from cv2 import cvtColor, COLOR_BGR2RGB, VideoCapture
import numpy as np
from matplotlib import pyplot as plt


# https://gist.github.com/mstfldmr/45d6e47bb661800b982c39d30215bc88
def cvshow(img):
    """convert and show
    not useful for our black and white images
    """
    cvtColor(img, COLOR_BGR2RGB)
    display(Image.fromarray(img))


def cvplt(frame, show=True):
    """ show with matplotlib """
    plt.imshow(frame)
    plt.xticks([]), plt.yticks([])  # no ticks
    if show:
        plt.show()

def cvplt_sub(frames, nx=1, ny=None):
    """ plot a few together. default to show all in a row
    @param frames: list of frames or
                   dict like {'title': frame, 'title2': frame}
    """
    if not ny:
        ny = len(frames)
    for i, f in enumerate(frames):
        plt.subplot(nx, ny, i+1)
        if type(frames) == list:
            cvplt(f, show=False)
        elif type(frames) == dict:
            plt.title(f)
            cvplt(frames[f], show=False)


def mkgray(img):
    "make 2d gray image from 3d color byaveraging all colors"
    gry = np.mean(img, axis=2)
    return gry


def avg_frame(f1, f2):
    return np.array((f1+f2)/2, dtype=int)


def animate_matrix(mat):
    """ animate a 3d matrix frame by frame """
    # from matplotlib import rc
    # rc('animation',html='html5')
    from matplotlib import animation
    im = plt.imshow(mat[:, :, 0])
    plt.title('init')
    f = plt.gcf()

    def upimg(i):
        plt.title(f'{i}')
        im.set_array(mat[:, :, i])
        return im,
    return animation.FuncAnimation(f, upimg, frames=mat.shape[2], blit=False)


class VidHelp:
    def __init__(self, vid_fname):
        self.vid_fname = vid_fname
        self.vs = VideoCapture(vid_fname)

    def get_frame(self, i, gray=False):
        "frame at pos"
        self.vs.set(1, i)
        img = self.vs.read()[1]
        if gray:
            img = mkgray(img)
        return img

    def get_frames(self, idxs):
        frames = [self.get_frame(i, gray=True) for i in idxs]
        return np.stack(frames, axis=2)

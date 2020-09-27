from PIL import Image
from IPython.display import display
from cv2 import cvtColor, COLOR_BGR2RGB
from matplotlib import pyplot as plt

# https://gist.github.com/mstfldmr/45d6e47bb661800b982c39d30215bc88
def cvshow(img):
    """convert and show
    not useful for our black and white images
    """
    cvtColor(img, COLOR_BGR2RGB)
    display(Image.fromarray(disp_img))

def cvplt(frame, show=True):
    """ show with matplotlib """
    plt.imshow(frame)
    plt.xticks([]), plt.yticks([])  # no ticks
    if show:
        plt.show()

def cvplt_sub(frames, nx, ny):
    """ plot a few together """
    for i, f in enumerate(frames):
        plt.subplot(nx, ny, i+1)
        cvplt(f, show=False)
    plt.show()



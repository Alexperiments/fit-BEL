import matplotlib.pyplot as plt
import numpy as np
import config

import os
import pickle
from Spectrum import Spectrum

file_path = 'examples/sample.fits'
a_v_extinction = 0.2

class Intervals_collector:
    def __init__(self, fig):
        self.continuum = intervals_dict['continuum']
        self.masks = []
        self.cid1 = fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid2 = fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid2 = fig.canvas.mpl_connect('close_event', self.on_close)
        self.continuum_mode = False
        self.masks_mode = False
        self.fig = fig
        self.ax = fig.gca()
        self.spans = []

    def on_key(self, event):
        if event.key == 'c':
            print("Seleziona intervalli per il continuo (max 4 punti); 'canc' per eliminare ultima selezione")
            self.continuum_mode = True
            self.masks_mode = False
        elif event.key == 'm':
            print("Seleziona coppie di punti per mascherare un intervallo; 'canc' per eliminare ultima selezione")
            self.continuum_mode = False
            self.masks_mode = True
        elif event.key == 'r':
            print("Premere 'c' per il continuo; 'm' per le maschere; 'f4' per terminare e salvare")
            self.continuum_mode = False
            self.masks_mode = False
        elif event.key == 'delete':
            try:
                if self.continuum_mode:
                    self.continuum.pop()
                    del self.ax.lines[1:]
                    self.update()
                elif self.masks_mode:
                    self.masks.pop()
                    self.masks.pop()
                    self.spans[-1].remove()
                    del self.spans[-1]
                    self.fig.canvas.draw()
                else: pass
            except(IndexError):
                print("Lista vuota!")
                pass
        elif event.key == 'n':
            del self.ax.lines[1:]
            intervals_dict['continuum'] = []
            self.continuum = []
            self.update()
        elif event.key == 'f4':
            if (len(self.continuum) == 4) and (len(self.masks)%2 == 0):
                intervals_dict['continuum'] = self.continuum
                intervals_dict['masks'] = self.masks
                name = intervals_dict['name'] + '.png'
                plt.savefig(config.PREPARATION_PLOTS+name, dpi=300)
                plt.close(self.fig)
            else:
                print(
                f"Punti continuo: {len(self.continuum)}/4\tPunti maschera: {len(self.masks)}"
                )

    def on_click(self, event):
        if self.continuum_mode:
            if len(self.continuum) < 4:
                self.continuum.append(event.xdata)
            else: print("Ho già 4 punti per il continuo, rimuoverne qualcuno!")
        elif self.masks_mode: self.masks.append(event.xdata)
        self.update()

    def on_close(self, event):
        del self.continuum
        del self.masks

    def update(self):
        for line in self.continuum:
            self.ax.axvline(line, color='green', ls='--')
        if (len(self.masks)%2 == 0) & (len(self.masks) != 0):
            self.spans.append(self.ax.axvspan(self.masks[-2], self.masks[-1], alpha=0.5, color='gray'))
        if len(self.continuum) == 4:
            m, q = continuum_fit(wl, fl, self.continuum)
            intervals_dict['m'] = m
            intervals_dict['q'] = q
            x_bin = np.arange(self.continuum[0], self.continuum[3], 1)
            self.ax.plot(x_bin, q + m*x_bin, color='red')
        self.fig.canvas.draw()


def continuum_fit(wl, flux, interval):
    mask = (
        ((wl >= interval[0]) &
        (wl < interval[1])) |
        ((wl >= interval[2]) &
        (wl < interval[3]))
    )
    m, q = np.polyfit(wl, flux, 1)
    return m, q


if __name__ == '__main__':
    file_path = 'examples/sample.fits'
    obj = Spectrum(file_path, redshift=3)
    wl, fl = obj.get_spectrum()

    intervals_dict = {
        'name': "Name",
        'continuum': config.CONTINUUM_INTERVALS,
        'masks': []
    }

    m, q = continuum_fit(wl, fl, intervals_dict['continuum'])
    intervals_dict['m'] = m
    intervals_dict['q'] = q

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)
    ax.plot(wl, fl, color='black', lw=0.5)
    ax.set_title(intervals_dict['name'])
    for xi in intervals_dict['continuum']:
        ax.axvline(xi, color='green', ls='--')
    for i in range(len(intervals_dict['masks']) // 2):
        ax.axvspan(intervals_dict['masks'][i * 2], intervals_dict['masks'][i * 2 + 1])
    x_bin = np.arange(intervals_dict['continuum'][0], intervals_dict['continuum'][3], 1)
    ax.plot(x_bin, intervals_dict['q'] + intervals_dict['m'] * x_bin, color='red')
    intervals = Intervals_collector(fig)
    plt.show()

    print(intervals_dict)
    # with open(os.path.join(file_path, 'pre_fit.pkl'), 'wb') as f:
    #     pickle.dump(intervals_dict, f)

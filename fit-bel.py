import os
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import config
import param
from fit import fit, gaussian_model

from Spectrum import Spectrum

parser = argparse.ArgumentParser(prog='fit-BEL',
                                    usage='%(prog)s path [options]',
                                    description='Estimate the AGN parameters from a single epoch spectrum',
                                    fromfile_prefix_chars='@')

parser.add_argument('Path', metavar='path', type=str, help='the path to spectrum file')
parser.add_argument('-z', '--redshift', type=float, required=True, help='redshift of the source')
parser.add_argument('-e', '--extinction', type=float, required=True, help='A_v parameter')
parser.add_argument('-o', '--output', type=str, help='optional output folder', default='output/')
parser.add_argument('-p', '--plot', type=str, help='optional output plot folder', default='figure/')


class InteractiveLineFit:
    def __init__(self, wl, flux, ivar, spectrum_dict,
                 obj_name='line fit', fit_model='gaussian_mixture', figsize=(15, 7)):
        self.q = None
        self.m = None
        self.wl = wl
        self.flux = flux
        self.ivar = ivar
        self.continuum_intervals = config.CONTINUUM_INTERVALS
        self.continuum_selection_lines = []
        self.continuum_fit_line = None
        self.masks = config.DEFAULT_MASKS
        self.spans = []
        self.fit_line = None
        self.fit_pars = None

        self.continuum_mode = False
        self.mask_mode = False
        self.fit_mode = False

        self.fit_model = fit_model

        self.continuum_fit()

        self.spectrum_dict = spectrum_dict
        self.spectrum_dict['name'] = obj_name
        self.spectrum_dict['continuum'] = self.continuum_intervals
        self.spectrum_dict['masks'] = self.masks
        self.spectrum_dict['m'] = self.m
        self.spectrum_dict['q'] = self.q

        self._init_plot(figsize)

        self.cid1 = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def _init_plot(self, figsize):
        plt.rcParams['keymap.fullscreen'].remove('f')
        plt.rcParams['keymap.save'].remove('s')
        plt.rcParams['keymap.back'].remove('c')
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.wl, self.flux, color='black', lw=0.5)
        self.ax.set_title(self.spectrum_dict['name'])

        self._draw_all()

        keys_legend = [
            r'[n]ew: cancel continuum intervals',
            r'[c]ontinuum: continuum selection mode',
            r'[m]ask: mask selection mode',
            r'[f]it: fit selection mode',
            r'[r]eset: exit selection mode',
            r'canc: cancel last entry',
            r'[s]save: save intervals']
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        plt.legend([extra] * len(keys_legend), keys_legend, loc='upper right', title='Keys')

    def _draw_all(self):
        for xi in self.continuum_intervals:
            self._plot_continuum_line(xi)
        for xi in self.masks:
            self._add_mask(xi)
        if len(self.continuum_intervals) == 4:
            self._add_fit_continuum()

    def _update_plot(self):
        self.fig.canvas.draw()

    # Continuum intervals

    def _add_continuum_point(self, xdata):
        if len(self.continuum_intervals) < 4:
            self.continuum_intervals.append(xdata)
            self._plot_continuum_line(xdata)
        else:
            print("Remove a continuum point. press 'canc' in continuum mode.")
        if len(self.continuum_intervals) == 4:
            self._add_fit_continuum()

    def _plot_continuum_line(self, xdata):
        self.continuum_selection_lines.append(self.ax.axvline(xdata, color='green', ls='--'))
        self._update_plot()

    def _cancel_last_continuum(self):
        try:
            self.continuum_intervals.pop()
            point = self.continuum_selection_lines.pop()
            self.ax.lines.remove(point)
            self._cancel_continuum_fit_line()
            self._update_plot()
        except IndexError:
            pass

    def _cancel_continuum_lines(self):
        for _ in range(4):
            self._cancel_last_continuum()
        self._cancel_continuum_fit_line()
        self._update_plot()

    # Continuum fit line

    def continuum_fit(self):
        continuum_mask = (
            ((self.wl >= self.continuum_intervals[0]) &
             (self.wl < self.continuum_intervals[1])) |
            ((self.wl >= self.continuum_intervals[2]) &
             (self.wl < self.continuum_intervals[3]))
        )
        wl = self.wl[continuum_mask]
        flux = self.flux[continuum_mask]
        self.m, self.q = np.polyfit(wl, flux, 1)

    def _add_fit_continuum(self):
        self.continuum_intervals.sort()
        self.continuum_fit()
        self._plot_continuum_fit_line()

    def _plot_continuum_fit_line(self):
        if not self.continuum_fit_line:
            x_bin = np.arange(self.continuum_intervals[0], self.continuum_intervals[3], 1)
            self.continuum_fit_line = self.ax.plot(x_bin, self.q + self.m * x_bin, color='red')
            self._update_plot()

    def _cancel_continuum_fit_line(self):
        if self.continuum_fit_line:
            self.ax.lines.remove(self.continuum_fit_line[0])
            self.continuum_fit_line = None

    # Masks

    def _add_mask(self, xdata):
        self.masks.append(xdata)
        if len(self.masks) % 2 == 0:
            self._plot_mask_span(self.masks[-2], xdata)

    def _plot_mask_span(self, x1, x2):
        self.spans.append(self.ax.axvspan(x1, x2, color='grey', alpha=0.5))
        self._update_plot()

    def _cancel_last_mask(self):
        if len(self.masks) % 2 == 1:
            self.masks.pop()
        try:
            self.masks.pop()
            self.masks.pop()
            span = self.spans.pop()
            self.ax.patches.remove(span)
            self._update_plot()
        except IndexError:
            pass

    def _mask_spectrum(self):
        wl = self.wl.copy()
        masks = sorted(self.masks)
        bool_mask = np.full(len(self.wl), True)
        for first, second in zip(masks[::2], masks[1::2]):
            bool_mask = bool_mask & (~((wl > first) & (wl < second)))
        return self.wl[bool_mask], self.flux[bool_mask], self.ivar[bool_mask]

    # Line fit

    def _fit_line(self, n_components, fit_model):
        if n_components in [1, 2, 3]:
            masked_wl, masked_flux, masked_ivar = self._mask_spectrum()
            x_bin = np.arange(self.continuum_intervals[0], self.continuum_intervals[3], 1)
            continuum_sub_flux = masked_flux - (self.q + self.m * masked_wl)
            self.fit_pars, _ = fit(masked_wl, continuum_sub_flux, masked_ivar, n_components, fit_model)
            fit_model = gaussian_model(x_bin, *self.fit_pars)  # TODO: generalize this call to accept more models
            continuum_add_model = fit_model + (self.q + self.m * x_bin)
            self._plot_fit_line(x_bin, continuum_add_model)
        else:
            pass

    def _plot_fit_line(self, x, y):
        self._cancel_fit()
        self.fit_line = self.ax.plot(x, y, c='blue')
        self._update_plot()

    def _cancel_fit(self):
        try:
            if self.fit_line:
                self.ax.lines.remove(self.fit_line[0])
                self.fit_line = None
                self._update_plot()
        except IndexError:
            pass

    def _reset_mode(self):
        self.fit_mode = False
        self.continuum_mode = False
        self.mask_mode = False

    def _save_and_exit(self):
        if (len(self.continuum_intervals) == 4) and (len(self.masks) % 2 == 0) and self.fit_line:
            self.spectrum_dict['continuum'] = self.continuum_intervals
            self.spectrum_dict['masks'] = self.masks
            self.spectrum_dict['fit_pars'] = self.fit_pars

            name = self.spectrum_dict['name'] + '.png'
            plt.savefig(plot_path + name, dpi=300)
            plt.close(self.fig)
        else:
            print("Error! Before saving you should:")
            if not len(self.continuum_intervals) == 4:
                print(f"Select four points for the continuum selection ({len(self.continuum_intervals)}/4)")
            elif not len(self.masks) % 2 == 0:
                print(f"Even the number of point for the mask selection ({len(self.masks)})")
            elif not self.fit_line:
                print("Fit the emission line (press \"f\" and choose the number of components)")

    def on_key(self, event):
        if event.key == 'c':
            if not self.continuum_mode:
                print("Seleziona intervalli per il continuo (max 4 punti); 'canc' per \
                eliminare ultima selezione")
            self._reset_mode()
            self.continuum_mode = True
        elif event.key == 'm':
            if not self.mask_mode:
                print("Seleziona coppie di punti per mascherare un intervallo; 'canc' \
                per eliminare ultima selezione")
            self._reset_mode()
            self.mask_mode = True
        elif event.key == 'f':
            if not self.fit_mode:
                print("Choose the number of components (max 3): ")
            self._reset_mode()
            self.fit_mode = True
        elif event.key == 'r':
            print("Premere 'c' per il continuo; 'm' per le maschere; 'f4' per \
            terminare e salvare")
            self._reset_mode()
        elif event.key == 'n':
            self._cancel_continuum_lines()
        elif event.key == 's':
            self._save_and_exit()
        elif event.key == 'delete':

            if self.continuum_mode:
                self._cancel_last_continuum()
            elif self.mask_mode:
                self._cancel_last_mask()
            elif self.fit_mode:
                self._cancel_fit()
            else:
                pass
        if self.fit_mode:
            self.fit_mode = True
            if event.key in ['1', '2', '3']:
                n_components = int(event.key)
            else:
                n_components = None
            self._fit_line(n_components, self.fit_model)

    def on_click(self, event):
        if self.continuum_mode:
            self._add_continuum_point(event.xdata)
        elif self.mask_mode:
            self._add_mask(event.xdata)

    def get_param_dict(self):
        return self.spectrum_dict

    def get_fit_param(self):
        return self.fit_pars


if __name__ == '__main__':
    args = parser.parse_args()
    file_path = args.Path
    redshift = args.redshift
    a_v_extinction = args.extinction
    output_path = args.output
    plot_path = args.plot

    obj = Spectrum(file_path, redshift=redshift)
    obj_name = obj.name
    wl = obj.wavelength
    fl = obj.flux
    ivar = obj.ivar

    spectrum_dict = {}

    interactive_plot = InteractiveLineFit(wl, fl, ivar, spectrum_dict, obj_name=obj_name)
    plt.show()

    par_dict = param.calc_params(spectrum_dict, redshift)

    print(par_dict)

    if output_path:
        with open(output_path + obj_name + '.txt', 'w') as convert_file:
            convert_file.write(json.dumps(par_dict))

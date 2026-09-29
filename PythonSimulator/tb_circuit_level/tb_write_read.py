import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, MolmemPlotter
from molmem_lib.sources import PulseSource

sim = MolmemSimulator()
sim.addCrossbarMatrix(device_type="default", rows=8, cols=8)

n_write = 16520
t_write = 80e-9
t_period = 160e-9
v_write = 0.9
v_read = 0.1

t_write_cycle = n_write * t_period
t_write_total = 8 * t_write_cycle
t_read_start = t_write_total + 1e-3
t_read_pattern = 2e-6
t_read_total = 4 * t_read_pattern
t_stop = t_read_start + t_read_total + 1e-3

clk = PulseSource(node=None, amp=1.0, period=t_period, width=t_write)
edges = [i * t_write_cycle for i in range(9)] + [
    t_read_start,
    t_read_start + t_read_pattern,
    t_read_start + 2*t_read_pattern,
    t_read_start + 3*t_read_pattern,
    t_read_start + 4*t_read_pattern
]

def build_row_func(i):
    def vrow(t, sources):
        if t < t_write_total:
            if (i * t_write_cycle) <= t < ((i + 1) * t_write_cycle):
                return sources[0].getVoltage(t) * v_write
            return 0.0
        elif t >= t_read_start and t < (t_read_start + t_read_total):
            t_read = t - t_read_start
            if t_read < t_read_pattern:
                return v_read if i in [0, 1, 3, 5, 7] else 0.0
            elif t_read < 2 * t_read_pattern:
                return v_read if i in [0, 2, 4, 6] else 0.0
            elif t_read < 3 * t_read_pattern:
                return v_read if i in [0, 2, 4, 6] else 0.0
            return 0.0
        return 0.0
    return vrow

def build_col_func(j):
    def vcol(t, sources):
        if t < t_write_total:
            if (j * t_write_cycle) <= t < ((j + 1) * t_write_cycle):
                return 0.0
            return sources[0].getVoltage(t) * v_write
        return 0.0
    return vcol

for r in range(1, 9):
    sim.addBSource(row=r, vFunc=build_row_func(r-1), sources=[clk], edge_points=edges)
    
for c in range(1, 9):
    sim.addBSource(col=c, vFunc=build_col_func(c-1), sources=[clk], edge_points=edges)

sim.run(tEnd=t_stop, min_dt=1e-12, tol=1e-3)

plotter = MolmemPlotter(sim)
plotter.createFigure("write_read", title="Write-Read Diagonal Identity Matrix", subplots=(2, 1))
plotter.addWaveform("write_read", 'Voltage', row=1, subplotIdx=0, color='blue', label='V(Row 1)')
plotter.addWaveform("write_read", 'Voltage', row=8, subplotIdx=0, color='orange', label='V(Row 8)')
plotter.addWaveform("write_read", 'Current', row=1, col=1, subplotIdx=1, color='red', label='I(1,1)')
plotter.addWaveform("write_read", 'Current', row=8, col=8, subplotIdx=1, color='green', label='I(8,8)')
plotter.formatAxis("write_read", 0)
plotter.formatAxis("write_read", 1)
plotter.saveFigure("write_read", "sim_results/tb_write_read.png")
plotter.showAll()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmem_lib import MolmemSimulator, MolmemPlotter
from molmem_lib.sources import PulseSource

sim = MolmemSimulator()
sim.addCrossbarMatrix(device_type="default", rows=8, cols=8)

t_full_pot = 2.6432e-3
t_half_pot = 1.3216e-3
t_quarter_dep = 0.6608e-3

t_dev0_end = t_full_pot
t_dev2_start = t_dev0_end + 100e-9
t_dev2_end = t_dev2_start + t_half_pot
t_dev5_start = t_dev2_end + 100e-9
t_dev5_pot_end = t_dev5_start + t_full_pot
t_dev5_end = t_dev5_pot_end + t_quarter_dep
t_stop = t_dev5_end + 1e-3

vref_pos_src = PulseSource(node=None, amp=0.9, period=160e-9, width=80e-9)
vref_neg_src = PulseSource(node=None, amp=-0.75, period=160e-9, width=60e-9)

def vrow_0_func(t, sources):
    return sources[0].getVoltage(t) if t < t_dev0_end else 0.0

def vrow_2_func(t, sources):
    return sources[0].getVoltage(t) if (t >= t_dev2_start and t < t_dev2_end) else 0.0

def vrow_5_func(t, sources):
    if t >= t_dev5_start and t < t_dev5_pot_end:
        return sources[0].getVoltage(t)
    elif t >= t_dev5_pot_end and t < t_dev5_end:
        return sources[1].getVoltage(t)
    return 0.0

edges = [t_dev0_end, t_dev2_start, t_dev2_end, t_dev5_start, t_dev5_pot_end, t_dev5_end]

sim.addBSource(row=1, vFunc=vrow_0_func, sources=[vref_pos_src], edge_points=edges)
sim.addDcSource(row=2, voltage=0.0)
sim.addBSource(row=3, vFunc=vrow_2_func, sources=[vref_pos_src], edge_points=edges)
sim.addDcSource(row=4, voltage=0.0)
sim.addDcSource(row=5, voltage=0.0)
sim.addBSource(row=6, vFunc=vrow_5_func, sources=[vref_pos_src, vref_neg_src], edge_points=edges)
sim.addDcSource(row=7, voltage=0.0)
sim.addDcSource(row=8, voltage=0.0)

for c in range(1, 9):
    sim.addDcSource(col=c, voltage=0.0)

sim.run(tEnd=t_stop, min_dt=1e-12, tol=1e-4)

plotter = MolmemPlotter(sim)
plotter.createFigure("crossbar_write", title="3x3 Crossbar V/2 Write Scheme", subplots=(2, 1))
plotter.addWaveform("crossbar_write", 'Voltage', row=1, subplotIdx=0, color='blue', label='V(0) - Full Potentiation')
plotter.addWaveform("crossbar_write", 'Voltage', row=3, subplotIdx=0, color='orange', label='V(2) - Half Potentiation')
plotter.addWaveform("crossbar_write", 'Voltage', row=6, subplotIdx=0, color='green', label='V(5) - Full Pot / Part Depot')
plotter.addWaveform("crossbar_write", 'State', row=1, col=1, subplotIdx=1, color='blue', label='Device (0,0) State')
plotter.addWaveform("crossbar_write", 'State', row=3, col=4, subplotIdx=1, color='orange', label='Device (2,3) State')
plotter.addWaveform("crossbar_write", 'State', row=6, col=8, subplotIdx=1, color='green', label='Device (5,7) State')
plotter.formatAxis("crossbar_write", 0, ylabel="Wordline Voltages (V)")
plotter.formatAxis("crossbar_write", 1, xlabel="Time (s)", ylabel="State (n)")
plotter.saveFigure("crossbar_write", "sim_results/tb_crossbar_write.png")
plotter.showAll()

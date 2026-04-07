## File to store run command formate to save time and track parameters

### !!! Do not Delete !!!

### For PointAbsorberSimulation.py


#### Calm sea, long waves (small motions, mostly for sanity checks)

python PointAbsorberSimulation.py --buoymass 500 --buoyradius 5 --waterdensity 1000 --waterdepth 100 --wavefrequency 0.08 --wavedirection 3.14159 --waveamplitude 1.0

#### Moderate sea, near-resonant heave (strong response, no PTO)

python PointAbsorberSimulation.py --buoymass 500 --buoyradius 5 --waterdensity 1000 --waterdepth 100 --wavefrequency 0.20 --wavedirection 3.14159 --waveamplitude 2.0


### For PTOPointAborberSimulation.py

#### Near-resonant sea, moderate PTO (noticeable reduction in heave amplitude)

python PTOPointAbsorberSimulation.py --buoymass 500 --buoyradius 5 --waterdensity 1000 --waterdepth 100 --wavefrequency 0.20 --wavedirection 3.14159 --waveamplitude 2.0 --cpto 1.0e5 --kpto 5.0e5 --visualize True --save True


### For Run.py

python Run.py --tspan 180 --nfreqcomponents 40 --peakperiod 12.0 --significantwaveheight 2.0 --buoymass 5000 --buoyradius 5 --waterdensity 1000 --wavedirection 3.14159 --cpto 1.0e5 --kpto 0.0 --seed 42


